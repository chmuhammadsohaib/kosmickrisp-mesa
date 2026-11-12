/*
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"

#include "kk_accel.h"

/* VkAccelerationStructureInstanceKHR */
struct vk_instance {
   float transform[12];
   uint32_t custom_index_and_mask;
   uint32_t sbt_offset_and_flags;
   uint64_t reference;
};

/* MTLIndirectAccelerationStructureInstanceDescriptor */
struct mtl_instance {
   float transform[12];
   uint32_t options;
   uint32_t mask;
   uint32_t intersection_function_table_offset;
   uint32_t user_id;
   uint64_t acceleration_structure_id;
};

/* Converts Vulkan instances to Metal's indirect instance descriptors, right
 * before a TLAS build. Both transforms are row-major 3x4. A Vulkan instance
 * references its BLAS by device address, which is the address of the slot
 * holding the BLAS's Metal resource ID (see kk_acceleration_structure.c). */
KERNEL(64)
libkk_convert_instances(global struct mtl_instance *dst, uint64_t src_addr,
                        uint32_t array_of_pointers, uint32_t count)
{
   uint id = cl_global_id.x;
   if (id >= count)
      return;

   global const struct vk_instance *src;
   if (array_of_pointers)
      src = (global const struct vk_instance *)((global const uint64_t *)src_addr)[id];
   else
      src = ((global const struct vk_instance *)src_addr) + id;

   struct mtl_instance out;
   for (uint i = 0; i < 12; i++)
      out.transform[i] = src->transform[i];

   uint32_t vk_flags = src->sbt_offset_and_flags >> 24;
   uint32_t options = 0;
   if (vk_flags & KK_VK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE)
      options |= KK_INSTANCE_OPTION_DISABLE_TRIANGLE_CULLING;
   bool ccw = KK_VK_FRONT_FACE_IS_CCW;
   if (vk_flags & KK_VK_INSTANCE_TRIANGLE_FLIP_FACING)
      ccw = !ccw;
   if (ccw)
      options |= KK_INSTANCE_OPTION_TRIANGLE_FRONT_FACING_WINDING_CCW;
   if (vk_flags & KK_VK_INSTANCE_FORCE_OPAQUE)
      options |= KK_INSTANCE_OPTION_OPAQUE;
   if (vk_flags & KK_VK_INSTANCE_FORCE_NO_OPAQUE)
      options |= KK_INSTANCE_OPTION_NON_OPAQUE;
   out.options = options;

   out.user_id = src->custom_index_and_mask & 0xffffff;
   out.intersection_function_table_offset = src->sbt_offset_and_flags & 0xffffff;

   /* A null reference makes the instance inactive */
   if (src->reference) {
      out.mask = src->custom_index_and_mask >> 24;
      out.acceleration_structure_id = *(global const uint64_t *)src->reference;
   } else {
      out.mask = 0;
      out.acceleration_structure_id = 0;
   }

   dst[id] = out;
}
