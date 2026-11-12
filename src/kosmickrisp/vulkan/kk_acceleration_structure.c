/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_acceleration_structure.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_rt_stats.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/libkk/kk_accel.h"

#include "util/stack_array.h"
#include "util/u_math.h"
#include "util/vma.h"

/* Acceleration structures map onto Metal acceleration structures, built and
 * traversed by the GPU. Mesa's software BVH builder is not used.
 *
 * - Metal places acceleration structures only in private heaps, at multiples
 *   of 16 KiB (MTL_AS_PLACEMENT_ALIGN_B). Acceleration structure storage
 *   buffers can use the private memory type, which is listed first, and are
 *   aligned to 16 KiB; reported sizes, compacted sizes included, are
 *   multiples of 16 KiB, so structures packed one after another in such a
 *   buffer are placed in it. Others are suballocated from a pool of private
 *   heaps owned by the device. Both kinds of heap stay resident, so creating
 *   or destroying a structure never changes the residency set.
 * - The device address is the GPU address of an 8-byte slot holding the
 *   structure's MTLResourceID. Shaders and TLAS instances get a handle from an
 *   address with one load.
 * - TLAS builds convert the Vulkan instances into Metal indirect instance
 *   descriptors at the start of the scratch buffer, so builds allocate nothing.
 */

/* Slots */

void
kk_device_init_as_slots(struct kk_device *dev)
{
   simple_mtx_init(&dev->as_slots.mutex, mtx_plain);
   util_dynarray_init(&dev->as_slots.bos, NULL);
   util_dynarray_init(&dev->as_slots.free, NULL);
   dev->as_slots.count = 0;
}

void
kk_device_finish_as_slots(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->as_slots.bos, struct kk_bo *, bo) {
      KK_RT_STATS_ADD(dev, KK_RT_STAT_SLOT_BOS, (uint64_t)-1);
      KK_RT_STATS_ADD(dev, KK_RT_STAT_SLOT_BYTES, -(*bo)->size_B);
      kk_destroy_bo(dev, *bo);
   }
   util_dynarray_fini(&dev->as_slots.bos);
   util_dynarray_fini(&dev->as_slots.free);
   simple_mtx_destroy(&dev->as_slots.mutex);
}

static uint64_t *
kk_as_slot_map(struct kk_device *dev, uint32_t slot, uint64_t *gpu_out)
{
   struct kk_bo *bo = *util_dynarray_element(
      &dev->as_slots.bos, struct kk_bo *, slot / KK_AS_SLOTS_PER_BO);
   uint32_t offset_B = (slot % KK_AS_SLOTS_PER_BO) * sizeof(uint64_t);

   if (gpu_out)
      *gpu_out = bo->gpu + offset_B;
   return (uint64_t *)((uint8_t *)bo->cpu + offset_B);
}

static VkResult
kk_as_slot_alloc(struct kk_device *dev, uint64_t resource_id,
                 uint32_t *slot_out, uint64_t *addr_out)
{
   VkResult result = VK_SUCCESS;
   uint32_t slot;

   simple_mtx_lock(&dev->as_slots.mutex);

   if (util_dynarray_num_elements(&dev->as_slots.free, uint32_t)) {
      slot = util_dynarray_pop(&dev->as_slots.free, uint32_t);
   } else {
      if (dev->as_slots.count % KK_AS_SLOTS_PER_BO == 0) {
         struct kk_bo *bo;
         result = kk_alloc_bo(dev, &dev->vk.base,
                              KK_AS_SLOTS_PER_BO * sizeof(uint64_t),
                              sizeof(uint64_t), &bo);
         if (result != VK_SUCCESS)
            goto out;
         kk_bo_set_label(bo, "acceleration structure slots");
         KK_RT_STATS_ADD(dev, KK_RT_STAT_SLOT_BOS, 1);
         KK_RT_STATS_ADD(dev, KK_RT_STAT_SLOT_BYTES, bo->size_B);
         util_dynarray_append(&dev->as_slots.bos, bo);
      }
      slot = dev->as_slots.count++;
   }

   *kk_as_slot_map(dev, slot, addr_out) = resource_id;
   *slot_out = slot;

out:
   simple_mtx_unlock(&dev->as_slots.mutex);
   return result;
}

static void
kk_as_slot_free(struct kk_device *dev, uint32_t slot)
{
   simple_mtx_lock(&dev->as_slots.mutex);
   *kk_as_slot_map(dev, slot, NULL) = 0u;
   util_dynarray_append(&dev->as_slots.free, slot);
   simple_mtx_unlock(&dev->as_slots.mutex);
}

/* Pool of private heaps for structures that can't be placed in their
 * buffer's memory */

#define KK_AS_CHUNK_SIZE_B (64ull * 1024 * 1024)
/* util_vma_heap doesn't hand out 0, so chunk offsets start here */
#define KK_AS_CHUNK_BASE MTL_AS_PLACEMENT_ALIGN_B

struct kk_as_chunk {
   struct kk_bo *bo;
   struct util_vma_heap vma;
   uint64_t used_B;
};

void
kk_device_init_as_pool(struct kk_device *dev)
{
   simple_mtx_init(&dev->as_pool.mutex, mtx_plain);
   util_dynarray_init(&dev->as_pool.chunks, NULL);

   dev->as_pool.refit_limit = debug_get_num_option("KK_RT_REFIT_LIMIT", 8);
   dev->as_pool.refit_min_prims =
      debug_get_num_option("KK_RT_REFIT_MIN_PRIMS", 2000);
}

static void
kk_as_chunk_destroy(struct kk_device *dev, struct kk_as_chunk *chunk)
{
   KK_RT_STATS_ADD(dev, KK_RT_STAT_POOL_CHUNKS, (uint64_t)-1);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_POOL_BYTES, -chunk->bo->size_B);
   util_vma_heap_finish(&chunk->vma);
   kk_destroy_bo(dev, chunk->bo);
   free(chunk);
}

void
kk_device_finish_as_pool(struct kk_device *dev)
{
   util_dynarray_foreach(&dev->as_pool.chunks, struct kk_as_chunk *, chunk)
      kk_as_chunk_destroy(dev, *chunk);
   util_dynarray_fini(&dev->as_pool.chunks);
   simple_mtx_destroy(&dev->as_pool.mutex);
}

static VkResult
kk_as_pool_alloc(struct kk_device *dev, struct kk_acceleration_structure *as)
{
   const uint64_t size_B = align64(as->vk.size, MTL_AS_PLACEMENT_ALIGN_B);
   struct kk_as_chunk *chunk = NULL;
   uint64_t addr = 0;

   simple_mtx_lock(&dev->as_pool.mutex);

   util_dynarray_foreach(&dev->as_pool.chunks, struct kk_as_chunk *, c) {
      addr = util_vma_heap_alloc(&(*c)->vma, size_B, MTL_AS_PLACEMENT_ALIGN_B);
      if (addr) {
         chunk = *c;
         break;
      }
   }

   if (!chunk) {
      chunk = calloc(1, sizeof(*chunk));
      if (!chunk) {
         simple_mtx_unlock(&dev->as_pool.mutex);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      VkResult result =
         kk_alloc_private_bo(dev, &dev->vk.base,
                             MAX2(KK_AS_CHUNK_SIZE_B, size_B), &chunk->bo);
      if (result != VK_SUCCESS) {
         free(chunk);
         simple_mtx_unlock(&dev->as_pool.mutex);
         return result;
      }
      kk_bo_set_label(chunk->bo, "acceleration structure pool");
      util_vma_heap_init(&chunk->vma, KK_AS_CHUNK_BASE, chunk->bo->size_B);
      util_dynarray_append(&dev->as_pool.chunks, chunk);
      KK_RT_STATS_ADD(dev, KK_RT_STAT_POOL_CHUNKS, 1);
      KK_RT_STATS_ADD(dev, KK_RT_STAT_POOL_BYTES, chunk->bo->size_B);

      addr = util_vma_heap_alloc(&chunk->vma, size_B, MTL_AS_PLACEMENT_ALIGN_B);
      assert(addr);
   }

   chunk->used_B += size_B;
   simple_mtx_unlock(&dev->as_pool.mutex);

   as->chunk = chunk;
   as->chunk_offset = addr - KK_AS_CHUNK_BASE;
   as->mtl_handle = mtl_heap_new_acceleration_structure(
      chunk->bo->mtl_handle, as->vk.size, as->chunk_offset);
   return as->mtl_handle ? VK_SUCCESS
                         : vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
}

static void
kk_as_pool_free(struct kk_device *dev, struct kk_acceleration_structure *as)
{
   const uint64_t size_B = align64(as->vk.size, MTL_AS_PLACEMENT_ALIGN_B);
   struct kk_as_chunk *chunk = as->chunk;

   simple_mtx_lock(&dev->as_pool.mutex);
   util_vma_heap_free(&chunk->vma, as->chunk_offset + KK_AS_CHUNK_BASE, size_B);
   chunk->used_B -= size_B;

   /* Keep the first chunk for the next structures; let others go once
    * empty, e.g. when a level unloads */
   if (!chunk->used_B &&
       *util_dynarray_element(&dev->as_pool.chunks, struct kk_as_chunk *, 0) !=
          chunk) {
      util_dynarray_delete_unordered(&dev->as_pool.chunks, struct kk_as_chunk *,
                                     chunk);
      kk_as_chunk_destroy(dev, chunk);
   }
   simple_mtx_unlock(&dev->as_pool.mutex);
}

/* Places the structure in its buffer's memory if that's private and the
 * offset is aligned for Metal */
static bool
kk_as_place(struct kk_acceleration_structure *as,
            const struct kk_buffer *buffer)
{
   if (!buffer || !buffer->bo || !buffer->bo->private_heap)
      return false;

   const uint64_t offset_B = buffer->bo_offset + as->vk.offset;
   if (offset_B % MTL_AS_PLACEMENT_ALIGN_B ||
       offset_B + as->vk.size > buffer->bo->size_B)
      return false;

   as->mtl_handle = mtl_heap_new_acceleration_structure(
      buffer->bo->mtl_handle, as->vk.size, offset_B);
   return as->mtl_handle != NULL;
}

/* Objects */

static void
kk_as_destroy(struct kk_device *dev, struct kk_acceleration_structure *as,
              const VkAllocationCallbacks *pAllocator)
{
   if (as->slot_addr)
      kk_as_slot_free(dev, as->slot);

   if (as->mtl_handle) {
      mtl_release(as->mtl_handle);
      KK_RT_STATS_ADD(dev, as->placed ? KK_RT_STAT_AS_PLACED_DESTROYED
                                      : KK_RT_STAT_AS_POOLED_DESTROYED,
                      1);
   }
   if (as->chunk)
      kk_as_pool_free(dev, as);

   vk_object_free(&dev->vk, pAllocator, as);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateAccelerationStructureKHR(
   VkDevice _device, const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkAccelerationStructureKHR *pAccelerationStructure)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(vk_buffer, buffer, pCreateInfo->buffer);

   struct kk_acceleration_structure *as =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*as),
                       VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
   if (!as)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   as->vk.buffer = buffer;
   as->vk.offset = pCreateInfo->offset;
   as->vk.size = pCreateInfo->size;

   as->placed = kk_as_place(as, container_of(buffer, struct kk_buffer, vk));
   if (!as->placed) {
      VkResult result = kk_as_pool_alloc(dev, as);
      if (result != VK_SUCCESS) {
         kk_as_destroy(dev, as, pAllocator);
         return result;
      }
   }
   KK_RT_STATS_ADD(dev, as->placed ? KK_RT_STAT_AS_PLACED_CREATED
                                   : KK_RT_STAT_AS_POOLED_CREATED,
                   1);

   as->resource_id = mtl_acceleration_structure_get_resource_id(as->mtl_handle);

   VkResult result =
      kk_as_slot_alloc(dev, as->resource_id, &as->slot, &as->slot_addr);
   if (result != VK_SUCCESS) {
      kk_as_destroy(dev, as, pAllocator);
      return result;
   }

   *pAccelerationStructure = kk_acceleration_structure_to_handle(as);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyAccelerationStructureKHR(
   VkDevice _device, VkAccelerationStructureKHR accelerationStructure,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   VK_FROM_HANDLE(kk_acceleration_structure, as, accelerationStructure);

   if (!as)
      return;

   kk_as_destroy(dev, as, pAllocator);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
kk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_acceleration_structure, as, pInfo->accelerationStructure);
   return as->slot_addr;
}

/* Build descriptions */

static uint32_t
kk_as_vertex_format(VkFormat format)
{
   /* MTLAttributeFormat values */
   switch (format) {
   case VK_FORMAT_R32G32_SFLOAT:
      return 29; /* Float2 */
   case VK_FORMAT_R32G32B32_SFLOAT:
      return 30; /* Float3 */
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 31; /* Float4 */
   case VK_FORMAT_R16G16_SFLOAT:
      return 25; /* Half2 */
   case VK_FORMAT_R16G16B16_SFLOAT:
      return 26; /* Half3 */
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 27; /* Half4 */
   case VK_FORMAT_R16G16_SNORM:
      return 22; /* Short2Normalized */
   case VK_FORMAT_R16G16B16A16_SNORM:
      return 24; /* Short4Normalized */
   case VK_FORMAT_R16G16_UNORM:
      return 19; /* UShort2Normalized */
   case VK_FORMAT_R16G16B16A16_UNORM:
      return 21; /* UShort4Normalized */
   case VK_FORMAT_R8G8_SNORM:
      return 10; /* Char2Normalized */
   case VK_FORMAT_R8G8B8A8_SNORM:
      return 12; /* Char4Normalized */
   case VK_FORMAT_R8G8_UNORM:
      return 7; /* UChar2Normalized */
   case VK_FORMAT_R8G8B8A8_UNORM:
      return 9; /* UChar4Normalized */
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return 41; /* UInt1010102Normalized */
   default:
      return 0;
   }
}

bool
kk_acceleration_structure_vertex_format_supported(VkFormat format)
{
   return kk_as_vertex_format(format) != 0;
}

static enum mtl_acceleration_structure_usage
kk_as_usage(VkBuildAccelerationStructureFlagsKHR flags)
{
   enum mtl_acceleration_structure_usage usage =
      MTL_ACCELERATION_STRUCTURE_USAGE_NONE;

   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_REFIT;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_BUILD;
   else if (flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_INTERSECTION;
   if (flags & VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR)
      usage |= MTL_ACCELERATION_STRUCTURE_USAGE_MINIMIZE_MEMORY;

   return usage;
}

static const VkAccelerationStructureGeometryKHR *
kk_as_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *info,
               uint32_t i)
{
   return info->pGeometries ? &info->pGeometries[i] : info->ppGeometries[i];
}

/* Bytes at the start of the scratch buffer that hold converted instances */
static uint64_t
kk_as_instance_area_size(uint32_t instance_count)
{
   return align64((uint64_t)instance_count * KK_MTL_INSTANCE_DESCRIPTOR_SIZE,
                  KK_AS_SCRATCH_ALIGNMENT);
}

/* Describes a build to Metal. With ranges NULL, only primitive counts from
 * max_counts are filled in, for size queries. */
static void
kk_as_fill_desc(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *ranges,
                const uint32_t *max_counts,
                struct mtl_acceleration_structure_geometry *geometries,
                struct mtl_acceleration_structure_desc *desc)
{
   *desc = (struct mtl_acceleration_structure_desc){
      .usage = kk_as_usage(info->flags),
   };

   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      desc->is_instance = true;
      desc->instance_count =
         info->geometryCount ? (ranges ? ranges[0].primitiveCount
                                       : max_counts[0])
                             : 0;
      if (ranges && desc->instance_count)
         desc->instance_addr = info->scratchData.deviceAddress;
      return;
   }

   desc->geometry_count = info->geometryCount;
   desc->geometries = geometries;

   for (uint32_t i = 0; i < info->geometryCount; i++) {
      const VkAccelerationStructureGeometryKHR *geo = kk_as_geometry(info, i);
      const VkAccelerationStructureBuildRangeInfoKHR *range =
         ranges ? &ranges[i] : NULL;
      struct mtl_acceleration_structure_geometry *out = &geometries[i];

      *out = (struct mtl_acceleration_structure_geometry){
         .opaque = geo->flags & VK_GEOMETRY_OPAQUE_BIT_KHR,
         .allow_duplicate_any_hit =
            !(geo->flags &
              VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR),
         .primitive_count = range ? range->primitiveCount : max_counts[i],
      };

      if (geo->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         const VkAccelerationStructureGeometryTrianglesDataKHR *tri =
            &geo->geometry.triangles;
         out->kind = MTL_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES;
         out->vertex_format = kk_as_vertex_format(tri->vertexFormat);
         out->vertex_stride_B = tri->vertexStride;
         out->index_size_B = tri->indexType == VK_INDEX_TYPE_UINT16 ? 2 : 4;

         if (!range)
            continue;

         /* Indices are relative to firstVertex, and Metal has no base vertex,
          * so the vertex buffer starts there. Without indices, primitiveOffset
          * is an offset into the vertex buffer. */
         const bool indexed = tri->indexType != VK_INDEX_TYPE_NONE_KHR;
         out->vertex_addr = tri->vertexData.deviceAddress +
                            (uint64_t)range->firstVertex * tri->vertexStride;
         if (indexed)
            out->index_addr =
               tri->indexData.deviceAddress + range->primitiveOffset;
         else
            out->vertex_addr += range->primitiveOffset;

         if (tri->transformData.deviceAddress)
            out->transform_addr =
               tri->transformData.deviceAddress + range->transformOffset;
      } else {
         assert(geo->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR);
         const VkAccelerationStructureGeometryAabbsDataKHR *aabbs =
            &geo->geometry.aabbs;
         out->kind = MTL_ACCELERATION_STRUCTURE_GEOMETRY_BOUNDING_BOXES;
         out->aabb_stride_B = aabbs->stride;
         if (range)
            out->aabb_addr = aabbs->data.deviceAddress + range->primitiveOffset;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(kk_device, dev, _device);

   STACK_ARRAY(struct mtl_acceleration_structure_geometry, geometries,
               pBuildInfo->geometryCount);
   struct mtl_acceleration_structure_desc desc;
   kk_as_fill_desc(pBuildInfo, NULL, pMaxPrimitiveCounts, geometries, &desc);

   struct mtl_acceleration_structure_sizes sizes;
   mtl_acceleration_structure_get_sizes(dev->mtl_handle, &desc, &sizes);
   STACK_ARRAY_FINISH(geometries);

   const uint64_t instance_area_B = kk_as_instance_area_size(desc.instance_count);

   /* Multiples of the placement alignment, so structures packed one after
    * another stay placeable */
   pSizeInfo->accelerationStructureSize =
      align64(sizes.acceleration_structure_size_B, MTL_AS_PLACEMENT_ALIGN_B);
   pSizeInfo->buildScratchSize =
      instance_area_B +
      align64(sizes.build_scratch_size_B, KK_AS_SCRATCH_ALIGNMENT);
   /* Updates of bottom-level structures may be done as builds (see
    * kk_as_rebuild_instead), so they get build scratch too */
   uint64_t update_scratch_B = sizes.refit_scratch_size_B;
   if (pBuildInfo->type != VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
      update_scratch_B = MAX2(update_scratch_B, sizes.build_scratch_size_B);
   pSizeInfo->updateScratchSize =
      instance_area_B + align64(update_scratch_B, KK_AS_SCRATCH_ALIGNMENT);
}

/* Commands */

/* Whether to build a bottom-level structure the app asked to update. Metal's
 * refit keeps the tree built for the original geometry, so traversal slows
 * as geometry moves away from it, refit after refit; small structures build
 * about as fast as they refit. */
static bool
kk_as_rebuild_instead(
   struct kk_cmd_buffer *cmd,
   const VkAccelerationStructureBuildGeometryInfoKHR *info,
   const VkAccelerationStructureBuildRangeInfoKHR *ranges,
   const struct kk_acceleration_structure *src)
{
   const struct kk_device *dev = kk_cmd_buffer_device(cmd);

   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
      return false;

   enum kk_rt_stat reason;
   uint64_t primitives = 0;
   for (uint32_t g = 0; g < info->geometryCount; g++)
      primitives += ranges[g].primitiveCount;

   if (primitives < (uint64_t)dev->as_pool.refit_min_prims)
      reason = KK_RT_STAT_REBUILDS_SMALL;
   else if (dev->as_pool.refit_limit > 0 &&
            src->refits >= (uint32_t)dev->as_pool.refit_limit)
      reason = KK_RT_STAT_REBUILDS_REFIT_LIMIT;
   else
      return false;

   if (unlikely(dev->rt_stats))
      cmd->rt_stats.counts[reason]++;
   return true;
}

static void
kk_as_count_builds(
   struct kk_cmd_buffer *cmd, uint32_t info_count,
   const VkAccelerationStructureBuildGeometryInfoKHR *infos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *range_infos)
{
   uint64_t *counts = cmd->rt_stats.counts;

   for (uint32_t i = 0; i < info_count; i++) {
      const bool update =
         infos[i].mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;

      uint64_t primitives = 0;
      for (uint32_t g = 0; g < infos[i].geometryCount; g++)
         primitives += range_infos[i][g].primitiveCount;

      if (infos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
         counts[update ? KK_RT_STAT_TLAS_UPDATES : KK_RT_STAT_TLAS_BUILDS]++;
         counts[update ? KK_RT_STAT_TLAS_UPDATE_INSTANCES
                       : KK_RT_STAT_TLAS_BUILD_INSTANCES] += primitives;
      } else {
         counts[update ? KK_RT_STAT_BLAS_UPDATES : KK_RT_STAT_BLAS_BUILDS]++;
         counts[update ? KK_RT_STAT_BLAS_UPDATE_PRIMS
                       : KK_RT_STAT_BLAS_BUILD_PRIMS] += primitives;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   const bool stats = kk_cmd_buffer_device(cmd)->rt_stats != NULL;
   int32_t stats_pair = -1;
   if (unlikely(stats)) {
      kk_as_count_builds(cmd, infoCount, pInfos, ppBuildRangeInfos);
      cmd->rt_stats.counts[KK_RT_STAT_BUILD_CALLS]++;
      stats_pair = kk_rt_stats_gpu_begin(cmd, KK_RT_STATS_GPU_AS);
   }

   /* Builds in one call are independent: convert every TLAS's instances
    * first, wait once, then encode all builds with no barrier between them.
    */
   for (uint32_t i = 0; i < infoCount; i++) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = &pInfos[i];
      if (info->type != VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ||
          !info->geometryCount || !ppBuildRangeInfos[i][0].primitiveCount)
         continue;

      const VkAccelerationStructureGeometryInstancesDataKHR *instances =
         &kk_as_geometry(info, 0)->geometry.instances;
      const uint32_t count = ppBuildRangeInfos[i][0].primitiveCount;
      libkk_convert_instances(cmd, kk_grid_1d(count), false,
                              info->scratchData.deviceAddress,
                              instances->data.deviceAddress +
                                 ppBuildRangeInfos[i][0].primitiveOffset,
                              instances->arrayOfPointers, count);
   }

   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_barrier_after_encoder_stages(
      encoder, MTL_STAGE_DISPATCH | MTL_STAGE_BLIT, MTL_STAGE_ACCELERATION_STRUCTURE);

   for (uint32_t i = 0; i < infoCount; i++) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = &pInfos[i];
      VK_FROM_HANDLE(kk_acceleration_structure, dst,
                     info->dstAccelerationStructure);

      STACK_ARRAY(struct mtl_acceleration_structure_geometry, geometries,
                  info->geometryCount);
      struct mtl_acceleration_structure_desc desc;
      kk_as_fill_desc(info, ppBuildRangeInfos[i], NULL, geometries, &desc);

      const uint64_t scratch_addr = info->scratchData.deviceAddress +
                                    kk_as_instance_area_size(desc.instance_count);

      VK_FROM_HANDLE(kk_acceleration_structure, src,
                     info->srcAccelerationStructure);
      bool update = info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      if (update &&
          kk_as_rebuild_instead(cmd, info, ppBuildRangeInfos[i], src))
         update = false;
      dst->refits = update ? src->refits + 1 : 0;

      if (update) {
         mtl_refit_acceleration_structure(
            encoder, src->mtl_handle, src == dst ? NULL : dst->mtl_handle,
            &desc, scratch_addr, 0u);
      } else {
         mtl_build_acceleration_structure(encoder, dst->mtl_handle, &desc,
                                          scratch_addr, 0u);
      }

      STACK_ARRAY_FINISH(geometries);
   }

   if (unlikely(stats))
      kk_rt_stats_gpu_end(cmd, stats_pair);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(kk_acceleration_structure, dst, pInfo->dst);

   const bool stats = kk_cmd_buffer_device(cmd)->rt_stats != NULL;
   int32_t stats_pair = -1;
   if (unlikely(stats)) {
      cmd->rt_stats.counts[pInfo->mode ==
                                 VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR
                              ? KK_RT_STAT_AS_COMPACTIONS
                              : KK_RT_STAT_AS_COPIES]++;
      stats_pair = kk_rt_stats_gpu_begin(cmd, KK_RT_STATS_GPU_AS);
   }

   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   dst->refits = src->refits;

   /* A plain Metal copy needs a destination as large as the source object.
    * Vulkan lets a clone of a compacted structure go into a buffer of the
    * compacted size (idTech does this), so clones into a smaller destination
    * compact again, which yields the same structure in the space it needs. */
   if (pInfo->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ||
       dst->vk.size < src->vk.size)
      mtl_copy_and_compact_acceleration_structure(encoder, src->mtl_handle,
                                                  dst->mtl_handle);
   else
      mtl_copy_acceleration_structure(encoder, src->mtl_handle,
                                      dst->mtl_handle);

   if (unlikely(stats))
      kk_rt_stats_gpu_end(cmd, stats_pair);
}

/* Serialized structures start with this header, as the spec requires */
struct kk_as_serialized_header {
   uint8_t driver_uuid[VK_UUID_SIZE];
   uint8_t compat_uuid[VK_UUID_SIZE];
   uint64_t serialized_size_B;
   uint64_t deserialized_size_B;
   uint64_t handle_count;
};

uint64_t
kk_acceleration_structure_serialized_size(
   const struct kk_acceleration_structure *as)
{
   return sizeof(struct kk_as_serialized_header);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_acceleration_structure, src, pInfo->src);
   struct kk_physical_device *pdev =
      kk_device_physical(kk_cmd_buffer_device(cmd));

   /* Metal can't read an acceleration structure back as bytes, so only the
    * header is written for now; deserializing then yields an empty structure.
    * TODO_KOSMICKRISP: record the build inputs and rebuild on deserialize. */
   mesa_logw_once("KosmicKrisp: acceleration structure serialization writes "
                  "only the header");

   struct kk_as_serialized_header header = {
      .serialized_size_B = sizeof(header),
      .deserialized_size_B = src->vk.size,
      .handle_count = 0,
   };
   memcpy(header.driver_uuid, pdev->vk.properties.driverUUID, VK_UUID_SIZE);
   memcpy(header.compat_uuid, pdev->vk.properties.driverUUID, VK_UUID_SIZE);

   const uint32_t *words = (const uint32_t *)&header;
   for (uint32_t i = 0; i < sizeof(header) / sizeof(uint32_t); i++)
      kk_cmd_write(cmd, (struct libkk_imm_write){
                           pInfo->dst.deviceAddress + i * sizeof(uint32_t),
                           words[i]});
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   mesa_logw_once("KosmicKrisp: acceleration structure deserialization is not "
                  "supported yet");
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   const uint8_t *uuid = kk_device_physical(dev)->vk.properties.driverUUID;

   const bool compatible =
      memcmp(pVersionInfo->pVersionData, uuid, VK_UUID_SIZE) == 0 &&
      memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE, uuid, VK_UUID_SIZE) == 0;

   *pCompatibility = compatible
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/* Host commands. accelerationStructureHostCommands is false, so applications
 * must not call these, but they have to resolve: games look them up anyway
 * (DOOM Eternal quits when vkBuildAccelerationStructuresKHR is missing). */

static VkResult
kk_as_host_command_unsupported(struct kk_device *dev, const char *name)
{
   mesa_logw_once("KosmicKrisp: %s called, but acceleration structure host "
                  "commands aren't supported",
                  name);
   return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return kk_as_host_command_unsupported(dev,
                                         "vkBuildAccelerationStructuresKHR");
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return kk_as_host_command_unsupported(dev, "vkCopyAccelerationStructureKHR");
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return kk_as_host_command_unsupported(
      dev, "vkCopyAccelerationStructureToMemoryKHR");
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return kk_as_host_command_unsupported(
      dev, "vkCopyMemoryToAccelerationStructureKHR");
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   return kk_as_host_command_unsupported(
      dev, "vkWriteAccelerationStructuresPropertiesKHR");
}
