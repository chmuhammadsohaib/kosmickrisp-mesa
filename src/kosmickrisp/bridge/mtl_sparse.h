/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_SPARSE_H
#define MTL_SPARSE_H 1

#include "mtl_types.h"

#include <inttypes.h>
#include <stdbool.h>

/* Placement sparse resources map 16 KiB pages of placement heaps. Metal only
 * allows private storage for them. */
#define MTL_SPARSE_PAGE_SIZE_B (16u * 1024u)

struct kk_image_layout;

/* A private placement heap sparse resources can map pages of */
mtl_heap *mtl_new_sparse_backing_heap(mtl_device *device, uint64_t size_B);

mtl_buffer *mtl_new_placement_sparse_buffer(mtl_device *device,
                                            uint64_t size_B);

mtl_texture *
mtl_new_placement_sparse_texture(mtl_device *device,
                                 const struct kk_image_layout *layout);

/* Size of one sparse tile of the layout's texture type and format, in texels */
struct mtl_size mtl_placement_sparse_tile_size(mtl_device *device,
                                     const struct kk_image_layout *layout);

uint64_t mtl_texture_tail_size(mtl_texture *texture);

struct mtl_sparse_buffer_mapping {
   bool map; /* false unmaps */
   uint64_t first_page;
   uint64_t page_count;
   uint64_t heap_page; /* first heap page when mapping */
};

/* Region in tiles. For the mip tail level, origin.y is 0 and size.y is 1. */
struct mtl_sparse_texture_mapping {
   bool map;
   struct mtl_origin origin;
   struct mtl_size size;
   uint32_t level;
   uint32_t slice;
   uint64_t heap_page;
};

void mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count);

void mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count);

#endif /* MTL_SPARSE_H */
