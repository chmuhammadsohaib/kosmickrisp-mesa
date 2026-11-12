/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_sparse.h"

mtl_heap *
mtl_new_sparse_backing_heap(mtl_device *device, uint64_t size_B)
{
   return NULL;
}

mtl_buffer *
mtl_new_placement_sparse_buffer(mtl_device *device, uint64_t size_B)
{
   return NULL;
}

mtl_texture *
mtl_new_placement_sparse_texture(mtl_device *device,
                                 const struct kk_image_layout *layout)
{
   return NULL;
}

struct mtl_size
mtl_placement_sparse_tile_size(mtl_device *device, const struct kk_image_layout *layout)
{
   return (struct mtl_size){1, 1, 1};
}


uint64_t
mtl_texture_tail_size(mtl_texture *texture)
{
   return 0u;
}

void
mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count)
{
}

void
mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count)
{
}
