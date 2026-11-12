/*
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_ACCEL_H
#define KK_ACCEL_H

/* MTLAccelerationStructureInstanceOptions */
#define KK_INSTANCE_OPTION_DISABLE_TRIANGLE_CULLING           (1u << 0)
#define KK_INSTANCE_OPTION_TRIANGLE_FRONT_FACING_WINDING_CCW (1u << 1)
#define KK_INSTANCE_OPTION_OPAQUE                             (1u << 2)
#define KK_INSTANCE_OPTION_NON_OPAQUE                         (1u << 3)

/* VkGeometryInstanceFlagBitsKHR, which libcl doesn't define */
#define KK_VK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE (1u << 0)
#define KK_VK_INSTANCE_TRIANGLE_FLIP_FACING         (1u << 1)
#define KK_VK_INSTANCE_FORCE_OPAQUE                 (1u << 2)
#define KK_VK_INSTANCE_FORCE_NO_OPAQUE              (1u << 3)

/* Whether Vulkan's front-facing winding, without FLIP_FACING, is Metal's
 * counter-clockwise. It isn't: FLIP_FACING is an alias of
 * TRIANGLE_FRONT_COUNTERCLOCKWISE, so Vulkan defaults to clockwise like Metal.
 * The face-culling conformance tests have the final word. */
#define KK_VK_FRONT_FACE_IS_CCW false

#define KK_MTL_INSTANCE_DESCRIPTOR_SIZE 72u

#endif /* KK_ACCEL_H */
