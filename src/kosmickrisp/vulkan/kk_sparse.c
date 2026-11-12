/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_sparse.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_device.h"
#include "kk_device_memory.h"
#include "kk_format.h"
#include "kk_image.h"
#include "kk_physical_device.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "vk_format.h"

/* Sparse resources are Metal placement sparse buffers and textures. They map
 * 16 KiB pages of private placement heaps, the memory of the sparse memory
 * type (kk_alloc_private_bo), with MTL4CommandQueue mapping updates, which
 * vkQueueBindSparse issues in order with the queue's other work.
 *
 * A sparse image's opaque address space, used by opaque binds and its memory
 * requirements, holds each array layer in turn: the tiles of every level
 * before the mip tail in X, Y, Z order, then the layer's mip tail. */

static uint32_t
div_round_up(uint32_t value, uint32_t divisor)
{
   return (value + divisor - 1) / divisor;
}

static bool
kk_sparse_image_supported(const struct kk_physical_device *pdev,
                          VkFormat format, VkImageType type,
                          VkSampleCountFlagBits samples, VkImageTiling tiling)
{
   if (!pdev->settings.sparse_enabled || type != VK_IMAGE_TYPE_2D ||
       samples != VK_SAMPLE_COUNT_1_BIT || tiling != VK_IMAGE_TILING_OPTIMAL)
      return false;

   const enum pipe_format p_format = vk_format_to_pipe_format(format);
   return p_format != PIPE_FORMAT_NONE && kk_get_va_format(p_format) &&
          !util_format_is_depth_or_stencil(p_format) &&
          vk_format_get_plane_count(format) == 1;
}

bool
kk_sparse_image_create_supported(const struct kk_physical_device *pdev,
                                 const VkPhysicalDeviceImageFormatInfo2 *info)
{
   return kk_sparse_image_supported(pdev, info->format, info->type,
                                    VK_SAMPLE_COUNT_1_BIT, info->tiling);
}

void
kk_sparse_image_format_properties(
   struct kk_physical_device *pdev,
   const VkPhysicalDeviceSparseImageFormatInfo2 *info, uint32_t *count,
   VkSparseImageFormatProperties2 *properties)
{
   if (!kk_sparse_image_supported(pdev, info->format, info->type,
                                  info->samples, info->tiling)) {
      *count = 0;
      return;
   }

   if (!properties) {
      *count = 1;
      return;
   }
   if (*count == 0)
      return;
   *count = 1;

   struct kk_image_layout layout = {
      .type = MTL_TEXTURE_TYPE_2D,
      .sample_count_sa = 1,
   };
   layout.format.mtl =
      kk_get_va_format(vk_format_to_pipe_format(info->format))->mtl_pixel_format;
   const struct mtl_size tile = mtl_placement_sparse_tile_size(pdev->mtl_dev_handle, &layout);

   properties->properties = (VkSparseImageFormatProperties){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .imageGranularity = {tile.x, tile.y, tile.z},
      .flags = 0,
   };
}

VkResult
kk_image_init_sparse(struct kk_device *dev, struct kk_image *image)
{
   struct kk_image_plane *plane = &image->planes[0];
   struct kk_image_sparse *sparse = &image->sparse_info;

   plane->mtl_handle =
      mtl_new_placement_sparse_texture(dev->mtl_handle, &plane->layout);
   if (!plane->mtl_handle)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   kk_device_add_buffer_to_residency_set(dev, plane->mtl_handle);
   image->sparse = true;

   sparse->tile = mtl_placement_sparse_tile_size(dev->mtl_handle, &plane->layout);
   sparse->first_tail_level =
      MIN2(mtl_texture_first_mipmap_in_tail(plane->mtl_handle),
           plane->layout.levels);
   sparse->tail_pages =
      sparse->first_tail_level < plane->layout.levels
         ? div_round_up(mtl_texture_tail_size(plane->mtl_handle),
                        MTL_SPARSE_PAGE_SIZE_B)
         : 0;

   sparse->layer_pages = 0;
   for (uint32_t level = 0; level < sparse->first_tail_level; level++) {
      const uint32_t width = u_minify(plane->layout.width_px, level);
      const uint32_t height = u_minify(plane->layout.height_px, level);
      sparse->level_tiles[level] = (struct mtl_size){
         div_round_up(width, sparse->tile.x),
         div_round_up(height, sparse->tile.y),
         1,
      };
      sparse->layer_pages +=
         sparse->level_tiles[level].x * sparse->level_tiles[level].y;
   }
   sparse->layer_pages += sparse->tail_pages;

   return VK_SUCCESS;
}

void
kk_image_finish_sparse(struct kk_device *dev, struct kk_image *image)
{
   if (image->sparse && image->planes[0].mtl_handle)
      kk_device_remove_buffer_from_residency_set(dev,
                                                 image->planes[0].mtl_handle);
}

uint64_t
kk_image_sparse_size_B(const struct kk_image *image)
{
   return (uint64_t)image->sparse_info.layer_pages *
          image->planes[0].layout.layers * MTL_SPARSE_PAGE_SIZE_B;
}

void
kk_image_sparse_memory_requirements(const struct kk_image *image,
                                    VkSparseImageMemoryRequirements *req)
{
   const struct kk_image_sparse *sparse = &image->sparse_info;
   const struct kk_image_layout *layout = &image->planes[0].layout;
   const uint64_t tail_offset_B =
      (uint64_t)(sparse->layer_pages - sparse->tail_pages) *
      MTL_SPARSE_PAGE_SIZE_B;

   *req = (VkSparseImageMemoryRequirements){
      .formatProperties = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .imageGranularity = {sparse->tile.x, sparse->tile.y, sparse->tile.z},
         .flags = 0,
      },
      .imageMipTailFirstLod = sparse->first_tail_level,
      .imageMipTailSize = (uint64_t)sparse->tail_pages * MTL_SPARSE_PAGE_SIZE_B,
      .imageMipTailOffset = tail_offset_B,
      .imageMipTailStride =
         layout->layers > 1
            ? (uint64_t)sparse->layer_pages * MTL_SPARSE_PAGE_SIZE_B
            : 0,
   };
}

/* Binds */

/* Unmapping ignores the heap, but Metal still needs one */
static mtl_heap *
kk_sparse_heap(struct kk_queue *queue, VkDeviceMemory memory)
{
   VK_FROM_HANDLE(kk_device_memory, mem, memory);
   return mem ? mem->bo->mtl_handle
              : kk_queue_device(queue)->sparse_unmap_bo->mtl_handle;
}

static void
kk_bind_sparse_buffer(struct kk_queue *queue,
                      const VkSparseBufferMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      mtl_heap *heap = kk_sparse_heap(queue, bind->memory);
      const struct mtl_sparse_buffer_mapping mapping = {
         .map = bind->memory != VK_NULL_HANDLE,
         .first_page = bind->resourceOffset / MTL_SPARSE_PAGE_SIZE_B,
         .page_count = div_round_up(bind->size, MTL_SPARSE_PAGE_SIZE_B),
         .heap_page = bind->memoryOffset / MTL_SPARSE_PAGE_SIZE_B,
      };
      mtl_command_queue_update_buffer_mappings(queue->mtl_handle,
                                               buffer->metal.handle, heap,
                                               &mapping, 1);
   }
}

/* Mapping of one page of a sparse image's opaque address space */
static struct mtl_sparse_texture_mapping
kk_image_opaque_page(const struct kk_image *image, uint64_t page)
{
   const struct kk_image_sparse *sparse = &image->sparse_info;
   struct mtl_sparse_texture_mapping mapping = {
      .size = {1, 1, 1},
      .slice = page / sparse->layer_pages,
   };

   uint64_t in_layer = page % sparse->layer_pages;
   for (uint32_t level = 0; level < sparse->first_tail_level; level++) {
      const struct mtl_size tiles = sparse->level_tiles[level];
      const uint64_t level_pages = (uint64_t)tiles.x * tiles.y;
      if (in_layer < level_pages) {
         mapping.level = level;
         mapping.origin = (struct mtl_origin){in_layer % tiles.x,
                                              in_layer / tiles.x, 0};
         return mapping;
      }
      in_layer -= level_pages;
   }

   /* Mip tail: Metal counts its tiles along X */
   mapping.level = sparse->first_tail_level;
   mapping.origin = (struct mtl_origin){in_layer, 0, 0};
   return mapping;
}

static void
kk_bind_sparse_image_opaque(struct kk_queue *queue,
                            const VkSparseImageOpaqueMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   struct util_dynarray mappings = UTIL_DYNARRAY_INIT;

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseMemoryBind *bind = &info->pBinds[i];
      mtl_heap *heap = kk_sparse_heap(queue, bind->memory);
      const uint64_t first = bind->resourceOffset / MTL_SPARSE_PAGE_SIZE_B;
      const uint64_t count = div_round_up(bind->size, MTL_SPARSE_PAGE_SIZE_B);

      util_dynarray_clear(&mappings);
      for (uint64_t p = 0; p < count; p++) {
         struct mtl_sparse_texture_mapping mapping =
            kk_image_opaque_page(image, first + p);
         mapping.map = bind->memory != VK_NULL_HANDLE;
         mapping.heap_page = bind->memoryOffset / MTL_SPARSE_PAGE_SIZE_B + p;
         util_dynarray_append(&mappings, mapping);
      }

      mtl_command_queue_update_texture_mappings(
         queue->mtl_handle, image->planes[0].mtl_handle, heap,
         util_dynarray_begin(&mappings),
         util_dynarray_num_elements(&mappings,
                                    struct mtl_sparse_texture_mapping));
   }

   util_dynarray_fini(&mappings);
}

static void
kk_bind_sparse_image(struct kk_queue *queue,
                     const VkSparseImageMemoryBindInfo *info)
{
   VK_FROM_HANDLE(kk_image, image, info->image);
   const struct mtl_size tile = image->sparse_info.tile;

   for (uint32_t i = 0; i < info->bindCount; i++) {
      const VkSparseImageMemoryBind *bind = &info->pBinds[i];
      mtl_heap *heap = kk_sparse_heap(queue, bind->memory);

      /* Metal assigns heap pages to the region's tiles in X, Y, Z order */
      const struct mtl_sparse_texture_mapping mapping = {
         .map = bind->memory != VK_NULL_HANDLE,
         .origin = {bind->offset.x / tile.x, bind->offset.y / tile.y,
                    bind->offset.z / tile.z},
         .size = {div_round_up(bind->extent.width, tile.x),
                  div_round_up(bind->extent.height, tile.y),
                  div_round_up(bind->extent.depth, tile.z)},
         .level = bind->subresource.mipLevel,
         .slice = bind->subresource.arrayLayer,
         .heap_page = bind->memoryOffset / MTL_SPARSE_PAGE_SIZE_B,
      };
      mtl_command_queue_update_texture_mappings(
         queue->mtl_handle, image->planes[0].mtl_handle, heap, &mapping, 1);
   }
}

void
kk_queue_bind_sparse(struct kk_queue *queue,
                     const struct vk_queue_submit *submit)
{
   for (uint32_t i = 0; i < submit->buffer_bind_count; i++)
      kk_bind_sparse_buffer(queue, &submit->buffer_binds[i]);
   for (uint32_t i = 0; i < submit->image_opaque_bind_count; i++)
      kk_bind_sparse_image_opaque(queue, &submit->image_opaque_binds[i]);
   for (uint32_t i = 0; i < submit->image_bind_count; i++)
      kk_bind_sparse_image(queue, &submit->image_binds[i]);
}
