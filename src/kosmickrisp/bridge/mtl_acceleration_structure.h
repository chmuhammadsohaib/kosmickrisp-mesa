/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_ACCELERATION_STRUCTURE_H
#define MTL_ACCELERATION_STRUCTURE_H 1

#include "mtl_types.h"

#include <inttypes.h>
#include <stdbool.h>

/* Values match MTLAccelerationStructureUsage */
enum mtl_acceleration_structure_usage {
   MTL_ACCELERATION_STRUCTURE_USAGE_NONE = 0,
   MTL_ACCELERATION_STRUCTURE_USAGE_REFIT = 1 << 0,
   MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_BUILD = 1 << 1,
   MTL_ACCELERATION_STRUCTURE_USAGE_EXTENDED_LIMITS = 1 << 2,
   MTL_ACCELERATION_STRUCTURE_USAGE_PREFER_FAST_INTERSECTION = 1 << 4,
   MTL_ACCELERATION_STRUCTURE_USAGE_MINIMIZE_MEMORY = 1 << 5,
};

/* Values match MTLAccelerationStructureInstanceOptions */
enum mtl_acceleration_structure_instance_options {
   MTL_INSTANCE_OPTION_NONE = 0,
   MTL_INSTANCE_OPTION_DISABLE_TRIANGLE_CULLING = 1 << 0,
   MTL_INSTANCE_OPTION_TRIANGLE_FRONT_FACING_WINDING_CCW = 1 << 1,
   MTL_INSTANCE_OPTION_OPAQUE = 1 << 2,
   MTL_INSTANCE_OPTION_NON_OPAQUE = 1 << 3,
};

/* Layout of MTLIndirectAccelerationStructureInstanceDescriptor */
struct mtl_indirect_instance_descriptor {
   float transform[12]; /* Row-major 3x4, see mtl_acceleration_structure.m */
   uint32_t options;
   uint32_t mask;
   uint32_t intersection_function_table_offset;
   uint32_t user_id;
   uint64_t acceleration_structure_id;
};

enum mtl_acceleration_structure_geometry_kind {
   MTL_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES,
   MTL_ACCELERATION_STRUCTURE_GEOMETRY_BOUNDING_BOXES,
};

/* One Vulkan geometry, already resolved to GPU addresses. Addresses may be 0
 * when only sizes are queried. */
struct mtl_acceleration_structure_geometry {
   enum mtl_acceleration_structure_geometry_kind kind;
   bool opaque;
   bool allow_duplicate_any_hit;
   uint32_t primitive_count;

   /* Triangles */
   uint64_t vertex_addr;
   uint64_t vertex_size_B;
   uint32_t vertex_stride_B;
   uint32_t vertex_format; /* MTLAttributeFormat */
   uint64_t index_addr;    /* 0 for non-indexed geometry */
   uint32_t index_size_B;  /* 2 or 4 */
   uint64_t transform_addr; /* 0 when there is no transform */

   /* Bounding boxes */
   uint64_t aabb_addr;
   uint32_t aabb_stride_B;
};

struct mtl_acceleration_structure_desc {
   bool is_instance;
   enum mtl_acceleration_structure_usage usage;

   /* Bottom level */
   uint32_t geometry_count;
   const struct mtl_acceleration_structure_geometry *geometries;

   /* Top level: an array of mtl_indirect_instance_descriptor */
   uint64_t instance_addr;
   uint32_t instance_count;
};

struct mtl_acceleration_structure_sizes {
   uint64_t acceleration_structure_size_B;
   uint64_t build_scratch_size_B;
   uint64_t refit_scratch_size_B;
};

void mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_desc *desc,
   struct mtl_acceleration_structure_sizes *sizes);

mtl_acceleration_structure *
mtl_new_acceleration_structure(mtl_device *device, uint64_t size_B);

uint64_t
mtl_acceleration_structure_get_resource_id(mtl_acceleration_structure *as);

void mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B);

/* dst may be NULL for an in-place refit */
void mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *src,
   mtl_acceleration_structure *dst,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B);

void mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                     mtl_acceleration_structure *src,
                                     mtl_acceleration_structure *dst);

void mtl_copy_and_compact_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *src,
   mtl_acceleration_structure *dst);

/* Writes the compacted size as a 64-bit value to addr */
void mtl_write_compacted_acceleration_structure_size(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   uint64_t addr);

#endif /* MTL_ACCELERATION_STRUCTURE_H */
