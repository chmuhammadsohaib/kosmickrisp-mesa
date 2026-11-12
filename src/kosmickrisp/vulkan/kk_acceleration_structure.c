/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_acceleration_structure.h"

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/libkk/kk_accel.h"

#include "util/stack_array.h"
#include "util/u_math.h"

/* Acceleration structures map onto Metal acceleration structures, built and
 * traversed by the GPU. Mesa's software BVH builder is not used.
 *
 * - Each structure is its own Metal allocation, kept resident through the
 *   device residency set. It can't live in its VkBuffer: Metal only places
 *   acceleration structures in private-storage heaps, and KosmicKrisp memory
 *   is shared.
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
   util_dynarray_foreach(&dev->as_slots.bos, struct kk_bo *, bo)
      kk_destroy_bo(dev, *bo);
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

/* Objects */

static void
kk_as_destroy(struct kk_device *dev, struct kk_acceleration_structure *as,
              const VkAllocationCallbacks *pAllocator)
{
   if (as->slot_addr)
      kk_as_slot_free(dev, as->slot);

   if (as->mtl_handle) {
      kk_device_remove_acceleration_structure_from_residency_set(
         dev, as->mtl_handle);
      mtl_release(as->mtl_handle);
   }

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

   as->mtl_handle = mtl_new_acceleration_structure(dev->mtl_handle, as->vk.size);
   if (!as->mtl_handle) {
      kk_as_destroy(dev, as, pAllocator);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   kk_device_add_acceleration_structure_to_residency_set(dev, as->mtl_handle);

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

   pSizeInfo->accelerationStructureSize =
      align64(sizes.acceleration_structure_size_B, KK_AS_SCRATCH_ALIGNMENT);
   pSizeInfo->buildScratchSize =
      instance_area_B +
      align64(sizes.build_scratch_size_B, KK_AS_SCRATCH_ALIGNMENT);
   pSizeInfo->updateScratchSize =
      instance_area_B +
      align64(sizes.refit_scratch_size_B, KK_AS_SCRATCH_ALIGNMENT);
}

/* Commands */

VKAPI_ATTR void VKAPI_CALL
kk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   /* Builds in one call are independent, so convert every TLAS's instances
    * first and wait once. */
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

      if (info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
         VK_FROM_HANDLE(kk_acceleration_structure, src,
                        info->srcAccelerationStructure);
         mtl_refit_acceleration_structure(
            encoder, src->mtl_handle, src == dst ? NULL : dst->mtl_handle,
            &desc, scratch_addr, 0u);
      } else {
         mtl_build_acceleration_structure(encoder, dst->mtl_handle, &desc,
                                          scratch_addr, 0u);
      }

      STACK_ARRAY_FINISH(geometries);
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(kk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(kk_acceleration_structure, dst, pInfo->dst);

   mtl_compute_encoder *encoder = cs_get_compute(cmd);

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
