/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_rt_stats.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/compiler/msl_rt.h"

#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "vk_common_entrypoints.h"
#include "vk_pipeline.h"

/* Ray tracing pipelines run on Metal's hardware traversal:
 *
 * - Raygen, miss, closest-hit and callable shaders are Metal visible functions
 *   (kk_shader.c, msl_rt.c), any-hit and intersection shaders intersection
 *   functions. One Metal compute pipeline per VkPipeline links them all, with
 *   kk_rt_dispatch as the kernel, and a visible function table holding them.
 * - Each visible function has a device-wide slot, its index in every table
 *   (0 means no shader), and pipeline-independent binary functions have the
 *   same function handle ID in every Metal pipeline. A group handle therefore
 *   depends only on the group's shaders, so a pipeline library and the
 *   pipelines linking it (VK_KHR_pipeline_library) give the same handles.
 * - The hit shader binding table is Metal's intersection function buffer: a
 *   record starts with the ID of the group's intersection function (its
 *   any-hit or intersection shader, or a default), which Metal calls on
 *   non-opaque or procedural candidates. Every handle has its shader's slot
 *   at MSL_RT_RECORD_FN_OFFSET, which raygen, traceRay and executeCallable use
 *   to call it.
 */

#define KK_RT_MAX_THREADS (16u * 1024u)
/* Free stacks kept for reuse, about the command buffers in flight */
#define KK_RT_MAX_FREE_STACKS 8u

/* Callable shaders can nest below the deepest ray */
#define KK_RT_CALLABLE_DEPTH 2u

struct kk_rt_link {
   mtl_compute_pipeline_state *pipeline;
   mtl_visible_function_table *vft;
   uint64_t vft_id;

   /* Bytes of ray tracing stack each thread needs */
   uint32_t stack_size_B;
};

void
kk_device_init_rt(struct kk_device *dev)
{
   simple_mtx_init(&dev->rt.mutex, mtx_plain);
   dev->rt.links = _mesa_pointer_hash_table_create(NULL);
   util_dynarray_init(&dev->rt.free_slots, NULL);
   dev->rt.next_slot = 1;

   simple_mtx_init(&dev->rt_stacks.mutex, mtx_plain);
   util_dynarray_init(&dev->rt_stacks.free, NULL);
}

static void
kk_rt_stack_destroy(struct kk_device *dev, struct kk_bo *bo)
{
   KK_RT_STATS_ADD(dev, KK_RT_STAT_STACKS_DESTROYED, 1);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_STACK_BYTES_DESTROYED, bo->size_B);
   kk_destroy_bo(dev, bo);
}

/* Called after the queue is finished, since its command buffers release
 * stacks */
void
kk_device_finish_rt_stacks(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->rt_stacks.free, struct kk_bo *, bo)
      kk_rt_stack_destroy(dev, *bo);
   util_dynarray_fini(&dev->rt_stacks.free);
   simple_mtx_destroy(&dev->rt_stacks.mutex);
}

/* A stack of at least size_B for a command buffer, which gives it back with
 * kk_rt_stack_release when it's reset, after the GPU is done with it. Stacks
 * are tens of MiB; making one per command buffer cost a heap allocation,
 * page clearing and residency set changes every frame. */
struct kk_bo *
kk_rt_stack_acquire(struct kk_device *dev, uint64_t size_B)
{
   struct kk_bo *found = NULL;

   simple_mtx_lock(&dev->rt_stacks.mutex);
   util_dynarray_foreach(&dev->rt_stacks.free, struct kk_bo *, bo) {
      if ((*bo)->size_B >= size_B &&
          (!found || (*bo)->size_B < found->size_B))
         found = *bo;
   }
   if (found) {
      util_dynarray_delete_unordered(&dev->rt_stacks.free, struct kk_bo *,
                                     found);
   }
   simple_mtx_unlock(&dev->rt_stacks.mutex);

   if (found)
      return found;

   /* Round up so launches of slightly different sizes share stacks */
   struct kk_bo *bo;
   if (kk_alloc_bo(dev, &dev->vk.base, align64(size_B, 4ull << 20), 256u,
                   &bo) != VK_SUCCESS)
      return NULL;
   kk_bo_set_label(bo, "ray tracing stack");
   KK_RT_STATS_ADD(dev, KK_RT_STAT_STACKS_CREATED, 1);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_STACK_BYTES_CREATED, bo->size_B);
   return bo;
}

void
kk_rt_stack_release(struct kk_device *dev, struct kk_bo *bo)
{
   simple_mtx_lock(&dev->rt_stacks.mutex);
   if (util_dynarray_num_elements(&dev->rt_stacks.free, struct kk_bo *) <
       KK_RT_MAX_FREE_STACKS) {
      util_dynarray_append(&dev->rt_stacks.free, bo);
      bo = NULL;
   }
   simple_mtx_unlock(&dev->rt_stacks.mutex);

   if (bo)
      kk_rt_stack_destroy(dev, bo);
}

static void
kk_rt_link_destroy(struct kk_device *dev, struct kk_rt_link *link)
{
   if (link->vft) {
      kk_device_remove_buffer_from_residency_set(dev, link->vft);
      mtl_release(link->vft);
      KK_RT_STATS_ADD(dev, KK_RT_STAT_VFTS_DESTROYED, 1);
   }
   if (link->pipeline)
      mtl_release(link->pipeline);
   free(link);
}

void
kk_device_finish_rt(struct kk_device *dev)
{
   hash_table_foreach(dev->rt.links, entry)
      kk_rt_link_destroy(dev, entry->data);
   _mesa_hash_table_destroy(dev->rt.links, NULL);
   util_dynarray_fini(&dev->rt.free_slots);

   if (dev->rt.default_triangle)
      mtl_release(dev->rt.default_triangle);
   if (dev->rt.default_box)
      mtl_release(dev->rt.default_box);
   if (dev->rt.kernel)
      mtl_release(dev->rt.kernel);
   if (dev->rt.library)
      mtl_release(dev->rt.library);
   simple_mtx_destroy(&dev->rt.mutex);
}

/* A function's handle ID, the same in every Metal pipeline linking it, read
 * from a pipeline that links only it. Called with dev->rt.mutex held. */
static uint64_t
kk_rt_probe_function_id(struct kk_device *dev, mtl_binary_function *function)
{
   mtl_compute_pipeline_state *pipeline =
      mtl_new_compute_pipeline_state_with_functions(
         dev->mtl_compiler_handle, dev->rt.kernel, &function, 1, 1);
   if (!pipeline)
      return 0;
   const uint64_t id = mtl_compute_pipeline_function_id(pipeline, function);
   mtl_release(pipeline);
   return id;
}

/* Called with dev->rt.mutex held */
static VkResult
kk_rt_init_device_objects(struct kk_device *dev)
{
   if (dev->rt.library)
      return VK_SUCCESS;

   struct kk_physical_device *pdev = kk_device_physical(dev);
   char *source = ralloc_asprintf(NULL,
                                  "#include <metal_stdlib>\n"
                                  "using namespace metal;\n%s%s",
                                  msl_rt_preamble, msl_rt_dispatch_source);
   dev->rt.library = mtl_new_library(
      dev->mtl_compiler_handle, source, pdev->info.msl_version,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   ralloc_free(source);
   if (!dev->rt.library)
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);

   dev->rt.kernel =
      mtl_new_library_function_descriptor(dev->rt.library, "kk_rt_dispatch");
   dev->rt.default_triangle = mtl_new_binary_function(
      dev->mtl_compiler_handle, dev->rt.library, "kk_rt_default_triangle");
   dev->rt.default_box = mtl_new_binary_function(
      dev->mtl_compiler_handle, dev->rt.library, "kk_rt_default_box");
   if (!dev->rt.kernel || !dev->rt.default_triangle || !dev->rt.default_box)
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);

   dev->rt.default_triangle_id =
      kk_rt_probe_function_id(dev, dev->rt.default_triangle);
   dev->rt.default_box_id = kk_rt_probe_function_id(dev, dev->rt.default_box);
   if (!dev->rt.default_triangle_id || !dev->rt.default_box_id)
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);

   return VK_SUCCESS;
}

VkResult
kk_rt_register_shader(struct kk_device *dev, struct kk_shader *shader)
{
   simple_mtx_lock(&dev->rt.mutex);
   VkResult result = kk_rt_init_device_objects(dev);

   if (result == VK_SUCCESS && shader->pipeline.rt.function) {
      shader->pipeline.rt.slot =
         util_dynarray_num_elements(&dev->rt.free_slots, uint32_t)
            ? util_dynarray_pop(&dev->rt.free_slots, uint32_t)
            : dev->rt.next_slot++;
   }

   if (result == VK_SUCCESS && shader->pipeline.rt.intersection) {
      shader->pipeline.rt.intersection_id =
         kk_rt_probe_function_id(dev, shader->pipeline.rt.intersection);
      if (!shader->pipeline.rt.intersection_id)
         result = vk_error(dev, VK_ERROR_INVALID_SHADER_NV);
   }

   simple_mtx_unlock(&dev->rt.mutex);
   return result;
}

void
kk_rt_unregister_shader(struct kk_device *dev, struct kk_shader *shader)
{
   if (!shader->pipeline.rt.slot)
      return;

   simple_mtx_lock(&dev->rt.mutex);
   util_dynarray_append(&dev->rt.free_slots, shader->pipeline.rt.slot);
   simple_mtx_unlock(&dev->rt.mutex);
   shader->pipeline.rt.slot = 0;
}

static struct kk_shader *
kk_rt_stage_shader(const struct vk_pipeline_stage *stage)
{
   return container_of(stage->shader, struct kk_shader, vk);
}

static VkResult
kk_rt_link_pipeline(struct kk_device *dev, struct vk_rt_pipeline *pipeline,
                    uint32_t max_recursion_depth)
{
   simple_mtx_lock(&dev->rt.mutex);
   VkResult result = kk_rt_init_device_objects(dev);
   simple_mtx_unlock(&dev->rt.mutex);
   if (result != VK_SUCCESS)
      return result;

   struct kk_rt_link *link = calloc(1, sizeof(*link));
   /* At most two functions per stage, plus the two defaults */
   mtl_binary_function **functions =
      malloc((2 + 2 * pipeline->stage_count) * sizeof(*functions));
   if (!link || !functions) {
      free(link);
      free(functions);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   /* Library stages are in pipeline->stages too */
   uint32_t function_count = 0;
   functions[function_count++] = dev->rt.default_triangle;
   functions[function_count++] = dev->rt.default_box;
   uint32_t max_out_B = 0, max_slot = 0;
   for (uint32_t i = 0; i < pipeline->stage_count; i++) {
      struct kk_shader *shader = kk_rt_stage_shader(&pipeline->stages[i]);
      if (shader->pipeline.rt.function)
         functions[function_count++] = shader->pipeline.rt.function;
      if (shader->pipeline.rt.intersection)
         functions[function_count++] = shader->pipeline.rt.intersection;
      max_out_B = MAX2(max_out_B, shader->info.rt.out_size_B);
      max_slot = MAX2(max_slot, shader->pipeline.rt.slot);
   }

   /* Each recursion level (raygen, traced rays, callables) has a frame and
    * its outgoing payloads; Metal's call stack nests the same way. */
   const uint32_t levels = 1 + max_recursion_depth + KK_RT_CALLABLE_DEPTH;
   link->stack_size_B = levels * (MSL_RT_FRAME_SIZE + max_out_B);

   link->pipeline = mtl_new_compute_pipeline_state_with_functions(
      dev->mtl_compiler_handle, dev->rt.kernel, functions, function_count,
      levels * 2 + 2);
   free(functions);
   if (!link->pipeline) {
      kk_rt_link_destroy(dev, link);
      return vk_error(dev, VK_ERROR_INVALID_SHADER_NV);
   }

   link->vft = mtl_new_visible_function_table(link->pipeline, max_slot + 1);
   if (!link->vft) {
      kk_rt_link_destroy(dev, link);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   kk_device_add_buffer_to_residency_set(dev, link->vft);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_VFTS_CREATED, 1);
   link->vft_id = mtl_visible_function_table_resource_id(link->vft);

   for (uint32_t i = 0; i < pipeline->stage_count; i++) {
      struct kk_shader *shader = kk_rt_stage_shader(&pipeline->stages[i]);
      if (shader->pipeline.rt.function)
         mtl_visible_function_table_set(link->vft, link->pipeline,
                                        shader->pipeline.rt.slot,
                                        shader->pipeline.rt.function);
   }

   simple_mtx_lock(&dev->rt.mutex);
   _mesa_hash_table_insert(dev->rt.links, &pipeline->base, link);
   simple_mtx_unlock(&dev->rt.mutex);
   return VK_SUCCESS;
}

static struct kk_rt_link *
kk_rt_get_link(struct kk_device *dev, const struct vk_pipeline *pipeline)
{
   simple_mtx_lock(&dev->rt.mutex);
   struct hash_entry *entry = _mesa_hash_table_search(dev->rt.links, pipeline);
   simple_mtx_unlock(&dev->rt.mutex);
   return entry ? entry->data : NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateRayTracingPipelinesKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache, uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(kk_device, dev, _device);

   VkResult result = vk_common_CreateRayTracingPipelinesKHR(
      _device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos,
      pAllocator, pPipelines);

   /* Libraries are only linked into the pipelines using them */
   for (uint32_t i = 0; i < createInfoCount; i++) {
      VK_FROM_HANDLE(vk_pipeline, pipeline, pPipelines[i]);
      if (!pipeline)
         continue;
      if (pipeline->flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
         if (unlikely(dev->rt_stats))
            kk_rt_stats_add(dev, KK_RT_STAT_LIBRARIES, 1);
         continue;
      }

      VkResult link_result = kk_rt_link_pipeline(
         dev, container_of(pipeline, struct vk_rt_pipeline, base),
         pCreateInfos[i].maxPipelineRayRecursionDepth);
      if (link_result != VK_SUCCESS) {
         vk_common_DestroyPipeline(_device, pPipelines[i], pAllocator);
         pPipelines[i] = VK_NULL_HANDLE;
         if (result == VK_SUCCESS)
            result = link_result;
      } else if (unlikely(dev->rt_stats)) {
         kk_rt_stats_add(dev, KK_RT_STAT_PIPELINES, 1);
      }
   }

   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyPipeline(VkDevice _device, VkPipeline _pipeline,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);

   if (pipeline &&
       pipeline->bind_point == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
      simple_mtx_lock(&dev->rt.mutex);
      struct hash_entry *entry =
         _mesa_hash_table_search(dev->rt.links, pipeline);
      if (entry) {
         kk_rt_link_destroy(dev, entry->data);
         _mesa_hash_table_remove(dev->rt.links, entry);
      }
      simple_mtx_unlock(&dev->rt.mutex);
   }

   vk_common_DestroyPipeline(_device, _pipeline, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindPipeline(VkCommandBuffer commandBuffer,
                   VkPipelineBindPoint pipelineBindPoint, VkPipeline _pipeline)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);

   if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      cmd->state.rt.pipeline = pipeline;

   vk_common_CmdBindPipeline(commandBuffer, pipelineBindPoint, _pipeline);
}

/* Libraries and the pipelines linking them get the same handles, since they
 * only depend on the group's shaders. */
VKAPI_ATTR VkResult VKAPI_CALL
kk_GetRayTracingShaderGroupHandlesKHR(VkDevice _device, VkPipeline _pipeline,
                                      uint32_t firstGroup, uint32_t groupCount,
                                      size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);

   /* Invalid usage, but games do it after a failed pipeline creation (DOOM
    * Eternal queries a null pipeline); fail instead of crashing. */
   if (!pipeline ||
       pipeline->bind_point != VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
      mesa_logw_once("KosmicKrisp: vkGetRayTracingShaderGroupHandlesKHR with "
                     "no ray tracing pipeline");
      return vk_error(dev, VK_ERROR_UNKNOWN);
   }

   struct vk_rt_pipeline *rt_pipeline =
      container_of(pipeline, struct vk_rt_pipeline, base);
   if ((uint64_t)firstGroup + groupCount > rt_pipeline->group_count ||
       dataSize < (size_t)groupCount * MSL_RT_HANDLE_SIZE)
      return vk_error(dev, VK_ERROR_UNKNOWN);

   for (uint32_t g = 0; g < groupCount; g++) {
      const struct vk_rt_shader_group *group =
         &rt_pipeline->groups[firstGroup + g];
      uint8_t *handle = (uint8_t *)pData + g * MSL_RT_HANDLE_SIZE;
      memset(handle, 0, MSL_RT_HANDLE_SIZE);

      uint64_t intersection_id = 0;
      uint32_t slot = 0;
      switch (group->type) {
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
         intersection_id = dev->rt.default_triangle_id;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
         intersection_id = dev->rt.default_box_id;
         break;
      default:
         break;
      }

      for (uint32_t s = 0; s < group->stage_count; s++) {
         const struct kk_shader *shader =
            kk_rt_stage_shader(&group->stages[s]);

         switch (group->stages[s].stage) {
         case MESA_SHADER_ANY_HIT:
            /* TODO_KOSMICKRISP: any-hit isn't run for procedural groups */
            if (group->type ==
                VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
               intersection_id = shader->pipeline.rt.intersection_id;
            break;
         case MESA_SHADER_INTERSECTION:
            intersection_id = shader->pipeline.rt.intersection_id;
            break;
         default:
            slot = shader->pipeline.rt.slot;
            break;
         }
      }

      memcpy(handle, &intersection_id, sizeof(intersection_id));
      memcpy(handle + MSL_RT_RECORD_FN_OFFSET, &slot, sizeof(slot));
   }

   return VK_SUCCESS;
}

static void
kk_cmd_trace_rays(struct kk_cmd_buffer *cmd,
                  const VkStridedDeviceAddressRegionKHR *raygen,
                  const VkStridedDeviceAddressRegionKHR *miss,
                  const VkStridedDeviceAddressRegionKHR *hit,
                  const VkStridedDeviceAddressRegionKHR *callable,
                  uint32_t width, uint32_t height, uint32_t depth,
                  uint64_t indirect_addr)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   const struct kk_rt_link *link =
      cmd->state.rt.pipeline ? kk_rt_get_link(dev, cmd->state.rt.pipeline)
                             : NULL;
   if (!link)
      return;

   const uint64_t launch = (uint64_t)width * height * depth;
   const uint32_t threads =
      indirect_addr ? KK_RT_MAX_THREADS
                    : (uint32_t)MIN2(launch, (uint64_t)KK_RT_MAX_THREADS);
   if (threads == 0)
      return;

   struct kk_rt_state *rt = &cmd->state.rt;
   const uint64_t stack_B = (uint64_t)threads * link->stack_size_B;
   if (rt->stack_size_B < stack_B) {
      struct kk_bo *bo = kk_rt_stack_acquire(dev, stack_B);
      if (!bo) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      util_dynarray_append(&cmd->rt_stacks, bo);
      rt->stack = (struct kk_ptr){.gpu = bo->gpu, .cpu = bo->cpu};
      rt->stack_size_B = bo->size_B;
   }

   struct kk_descriptor_state *desc = &rt->descriptors;
   if (desc->push_dirty)
      kk_cmd_buffer_flush_push_descriptors(cmd, desc);
   if (desc->root_dirty || !desc->root.addr)
      kk_upload_descriptor_root(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);

   struct kk_ptr args_ptr =
      kk_pool_alloc(cmd, sizeof(struct msl_rt_dispatch_args), 16u);
   if (!args_ptr.gpu)
      return;

   struct msl_rt_dispatch_args *args = args_ptr.cpu;
   *args = (struct msl_rt_dispatch_args){
      .root = desc->root.addr,
      .sampler_table = dev->samplers.table.bo->gpu,
      .vft = args_ptr.gpu + offsetof(struct msl_rt_dispatch_args, vft_id),
      .hit_sbt = hit->deviceAddress,
      .miss_sbt = miss->deviceAddress,
      .callable_sbt = callable->deviceAddress,
      .hit_sbt_size = hit->size,
      .raygen_record = raygen->deviceAddress,
      .stack_base = rt->stack.gpu,
      .hit_stride = hit->stride,
      .miss_stride = miss->stride,
      .callable_stride = callable->stride,
      .stack_size = link->stack_size_B,
      .launch_size = {width, height, depth},
      .indirect = indirect_addr,
      .vft_id = link->vft_id,
   };

   const bool stats = dev->rt_stats != NULL;
   int32_t stats_pair = -1;
   if (unlikely(stats)) {
      if (indirect_addr) {
         cmd->rt_stats.counts[KK_RT_STAT_TRACES_INDIRECT]++;
      } else {
         cmd->rt_stats.counts[KK_RT_STAT_TRACES]++;
         cmd->rt_stats.counts[KK_RT_STAT_RAYS] += launch;
      }
      stats_pair = kk_rt_stats_gpu_begin(cmd, KK_RT_STATS_GPU_TRACE);
   }

   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   /* Traces in a command buffer share the stack */
   mtl_barrier_after_encoder_stages(
      encoder, MTL_STAGE_DISPATCH | MTL_STAGE_ACCELERATION_STRUCTURE,
      MTL_STAGE_DISPATCH);

   mtl_compute_set_pipeline_state(encoder, link->pipeline);
   mtl_set_address(cmd->argument_table, args_ptr.gpu, 0u);
   mtl_dispatch_threads(encoder, (struct mtl_size){threads, 1, 1},
                        (struct mtl_size){64, 1, 1});
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);

   if (unlikely(stats))
      kk_rt_stats_gpu_end(cmd, stats_pair);

   /* Restore the compute root */
   mtl_set_address(cmd->argument_table, cmd->state.root_addr, 0u);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdTraceRaysKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_cmd_trace_rays(cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
                     pHitShaderBindingTable, pCallableShaderBindingTable,
                     width, height, depth, 0u);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_cmd_trace_rays(cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
                     pHitShaderBindingTable, pCallableShaderBindingTable, 0, 0,
                     0, indirectDeviceAddress);
}

/* KosmicKrisp tracks the bound pipeline itself (kk_CmdBindPipeline) and sizes
 * the stack from the pipeline's recursion depth at link time. */
void
kk_cmd_set_rt_state(struct vk_command_buffer *cmd_buffer,
                    struct vk_pipeline_layout *layout,
                    VkDeviceSize scratch_size, uint32_t ray_queries,
                    const uint8_t *dynamic_descriptor_offsets)
{
}

void
kk_cmd_set_stack_size(struct vk_command_buffer *cmd_buffer,
                      VkDeviceSize stack_size)
{
}
