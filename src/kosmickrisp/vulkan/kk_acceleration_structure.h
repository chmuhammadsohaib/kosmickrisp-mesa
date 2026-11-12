/*
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_ACCELERATION_STRUCTURE_H
#define KK_ACCELERATION_STRUCTURE_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_acceleration_structure.h"

#define KK_AS_SLOTS_PER_BO 8192u

/* Metal needs no alignment beyond this for scratch and instance data */
#define KK_AS_SCRATCH_ALIGNMENT 256u

struct kk_acceleration_structure {
   struct vk_acceleration_structure vk;

   mtl_acceleration_structure *mtl_handle;
   uint64_t resource_id;

   /* The structure's device address is the GPU address of an 8-byte slot
    * holding resource_id, so shaders and instances get a handle with one load.
    */
   uint32_t slot;
   uint64_t slot_addr;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_acceleration_structure, vk.base,
                               VkAccelerationStructureKHR,
                               VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)

static inline uint64_t
kk_acceleration_structure_resource_id(VkAccelerationStructureKHR handle)
{
   VK_FROM_HANDLE(kk_acceleration_structure, as, handle);
   return as ? as->resource_id : 0u;
}

bool kk_acceleration_structure_vertex_format_supported(VkFormat format);

uint64_t kk_acceleration_structure_serialized_size(
   const struct kk_acceleration_structure *as);

#endif /* KK_ACCELERATION_STRUCTURE_H */
