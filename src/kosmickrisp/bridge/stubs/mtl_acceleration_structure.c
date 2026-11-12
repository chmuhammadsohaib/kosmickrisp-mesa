/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_acceleration_structure.h"

#include <string.h>

void
mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_desc *desc,
   struct mtl_acceleration_structure_sizes *sizes)
{
   memset(sizes, 0, sizeof(*sizes));
}


mtl_acceleration_structure *
mtl_new_acceleration_structure(mtl_device *device, uint64_t size_B)
{
   return NULL;
}


uint64_t
mtl_acceleration_structure_get_resource_id(mtl_acceleration_structure *as)
{
   return 0u;
}

void
mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B)
{
}

void
mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *src,
   mtl_acceleration_structure *dst,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B)
{
}

void
mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                mtl_acceleration_structure *src,
                                mtl_acceleration_structure *dst)
{
}

void
mtl_copy_and_compact_acceleration_structure(mtl_compute_encoder *encoder,
                                            mtl_acceleration_structure *src,
                                            mtl_acceleration_structure *dst)
{
}

void
mtl_write_compacted_acceleration_structure_size(mtl_compute_encoder *encoder,
                                                mtl_acceleration_structure *as,
                                                uint64_t addr)
{
}
