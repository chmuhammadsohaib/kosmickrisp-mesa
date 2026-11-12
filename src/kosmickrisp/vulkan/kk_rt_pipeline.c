/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/compiler/msl_rt.h"

#include "util/hash_table.h"
#include "vk_common_entrypoints.h"
#include "vk_pipeline.h"

/* Ray tracing pipelines run on Metal's hardware traversal:
 *
 * - Raygen, miss, closest-hit and callable shaders are Metal visible functions
 *   (kk_shader.c, msl_rt.c), any-hit and intersection shaders intersection
 *   functions. One
 *   Metal compute pipeline per VkPipeline links them all, with
 *   kk_rt_dispatch as the kernel, and gets a visible function table holding
 *   them at indices 1 and up (0 means no shader).
 * - The hit shader binding table is Metal's intersection function buffer: a
 *   record starts with the handle of the group's intersection function (its
 *   any-hit or intersection shader, or a default), which Metal calls on
 *   non-opaque or procedural candidates. Function handles of pipeline-
 *   independent binary functions have the same ID in every pipeline.
 * - Every handle has its shader's table index at MSL_RT_RECORD_FN_OFFSET, which
 *   raygen, traceRay and executeCallable use to call it.
 */

#define KK_RT_MAX_THREADS (16u * 1024u)

/* Callable shaders can nest below the deepest ray */
#define KK_RT_CALLABLE_DEPTH 2u

struct kk_rt_link {
   mtl_compute_pipeline_state *pipeline;
   mtl_visible_function_table *vft;
   uint64_t vft_id;

   uint64_t default_triangle_id;
   uint64_t default_box_id;

   /* Indexed like vk_rt_pipeline::stages */
   uint32_t stage_count;
   struct kk_rt_link_stage {
      const struct vk_shader *shader;
      uint32_t vft_index;
      uint64_t intersection_id;
   } *stages;

   /* Bytes of ray tracing stack each thread needs */
   uint32_t stack_size_B;
};

void
kk_device_init_rt(struct kk_device *dev)
{
   simple_mtx_init(&dev->rt.mutex, mtx_plain);
   dev->rt.links = _mesa_pointer_hash_table_create(NULL);
}

static void
kk_rt_link_destroy(struct kk_device *dev, struct kk_rt_link *link)
{
   if (link->vft) {
      kk_device_remove_buffer_from_residency_set(dev, link->vft);
      mtl_release(link->vft);
   }
   if (link->pipeline)
      mtl_release(link->pipeline);
   free(link->stages);
   free(link);
}

void
kk_device_finish_rt(struct kk_device *dev)
{
   hash_table_foreach(dev->rt.links, entry)
      kk_rt_link_destroy(dev, entry->data);
   _mesa_hash_table_destroy(dev->rt.links, NULL);

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

   return VK_SUCCESS;
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
   if (!link)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   link->stage_count = pipeline->stage_count;
   link->stages = calloc(MAX2(pipeline->stage_count, 1),
                         sizeof(*link->stages));

   /* At most two functions per stage, plus the two defaults */
   const uint32_t max_functions = 2 + 2 * pipeline->stage_count;
   mtl_binary_function **functions =
      malloc(max_functions * sizeof(*functions));
   if (!link->stages || !functions) {
      free(functions);
      kk_rt_link_destroy(dev, link);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   uint32_t function_count = 0;
   functions[function_count++] = dev->rt.default_triangle;
   functions[function_count++] = dev->rt.default_box;
   uint32_t max_out_B = 0;
   for (uint32_t i = 0; i < pipeline->stage_count; i++) {
      struct kk_shader *shader = kk_rt_stage_shader(&pipeline->stages[i]);
      if (shader->pipeline.rt.function)
         functions[function_count++] = shader->pipeline.rt.function;
      if (shader->pipeline.rt.intersection)
         functions[function_count++] = shader->pipeline.rt.intersection;
      max_out_B = MAX2(max_out_B, shader->info.rt.out_size_B);
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

   link->vft = mtl_new_visible_function_table(link->pipeline,
                                              pipeline->stage_count + 1);
   if (!link->vft) {
      kk_rt_link_destroy(dev, link);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   kk_device_add_buffer_to_residency_set(dev, link->vft);
   link->vft_id = mtl_visible_function_table_resource_id(link->vft);

   link->default_triangle_id =
      mtl_compute_pipeline_function_id(link->pipeline, dev->rt.default_triangle);
   link->default_box_id =
      mtl_compute_pipeline_function_id(link->pipeline, dev->rt.default_box);

   for (uint32_t i = 0; i < pipeline->stage_count; i++) {
      struct kk_shader *shader = kk_rt_stage_shader(&pipeline->stages[i]);
      link->stages[i].shader = &shader->vk;
      if (shader->pipeline.rt.function) {
         link->stages[i].vft_index = i + 1;
         mtl_visible_function_table_set(link->vft, link->pipeline, i + 1,
                                        shader->pipeline.rt.function);
      }
      if (shader->pipeline.rt.intersection)
         link->stages[i].intersection_id = mtl_compute_pipeline_function_id(
            link->pipeline, shader->pipeline.rt.intersection);
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

   for (uint32_t i = 0; i < createInfoCount; i++) {
      VK_FROM_HANDLE(vk_pipeline, pipeline, pPipelines[i]);
      if (!pipeline || (pipeline->flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))
         continue;

      VkResult link_result = kk_rt_link_pipeline(
         dev, container_of(pipeline, struct vk_rt_pipeline, base),
         pCreateInfos[i].maxPipelineRayRecursionDepth);
      if (link_result != VK_SUCCESS) {
         vk_common_DestroyPipeline(_device, pPipelines[i], pAllocator);
         pPipelines[i] = VK_NULL_HANDLE;
         if (result == VK_SUCCESS)
            result = link_result;
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

static const struct kk_rt_link_stage *
kk_rt_link_find_stage(const struct kk_rt_link *link,
                      const struct vk_shader *shader)
{
   for (uint32_t i = 0; i < link->stage_count; i++) {
      if (link->stages[i].shader == shader)
         return &link->stages[i];
   }
   return NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetRayTracingShaderGroupHandlesKHR(VkDevice _device, VkPipeline _pipeline,
                                      uint32_t firstGroup, uint32_t groupCount,
                                      size_t dataSize, void *pData)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);
   struct vk_rt_pipeline *rt_pipeline =
      container_of(pipeline, struct vk_rt_pipeline, base);

   /* Libraries have no link; their handles only matter once linked into a
    * pipeline, which has its own. */
   const struct kk_rt_link *link = kk_rt_get_link(dev, pipeline);

   for (uint32_t g = 0; g < groupCount; g++) {
      const struct vk_rt_shader_group *group =
         &rt_pipeline->groups[firstGroup + g];
      uint8_t *handle = (uint8_t *)pData + g * MSL_RT_HANDLE_SIZE;
      memset(handle, 0, MSL_RT_HANDLE_SIZE);
      if (!link)
         continue;

      uint64_t intersection_id = 0;
      uint32_t vft_index = 0;
      switch (group->type) {
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
         intersection_id = link->default_triangle_id;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
         intersection_id = link->default_box_id;
         break;
      default:
         break;
      }

      for (uint32_t s = 0; s < group->stage_count; s++) {
         const struct kk_rt_link_stage *stage =
            kk_rt_link_find_stage(link, group->stages[s].shader);
         if (!stage)
            continue;

         switch (group->stages[s].stage) {
         case MESA_SHADER_ANY_HIT:
            /* TODO_KOSMICKRISP: any-hit isn't run for procedural groups */
            if (group->type ==
                VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
               intersection_id = stage->intersection_id;
            break;
         case MESA_SHADER_INTERSECTION:
            intersection_id = stage->intersection_id;
            break;
         default:
            vft_index = stage->vft_index;
            break;
         }
      }

      memcpy(handle, &intersection_id, sizeof(intersection_id));
      memcpy(handle + MSL_RT_RECORD_FN_OFFSET, &vft_index, sizeof(vft_index));
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
      rt->stack = kk_pool_alloc(cmd, stack_B, 256u);
      rt->stack_size_B = rt->stack.gpu ? stack_B : 0;
      if (!rt->stack.gpu)
         return;
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
