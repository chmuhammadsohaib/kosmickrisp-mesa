/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_sparse.h"

#include "kk_private.h"

#include <Metal/MTL4CommandQueue.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLHeap.h>
#include <Metal/MTLTexture.h>

mtl_heap *
mtl_new_sparse_backing_heap(mtl_device *device, uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor new] autorelease];
      descriptor.type = MTLHeapTypePlacement;
      descriptor.storageMode = MTLStorageModePrivate;
      descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
      descriptor.size = size_B;
      descriptor.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize16;
      return [dev newHeapWithDescriptor:descriptor];
   }
}

mtl_buffer *
mtl_new_placement_sparse_buffer(mtl_device *device, uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newBufferWithLength:size_B
                              options:MTLResourceStorageModePrivate |
                                      MTLResourceHazardTrackingModeUntracked
              placementSparsePageSize:MTLSparsePageSize16];
   }
}


uint64_t
mtl_texture_tail_size(mtl_texture *texture)
{
   @autoreleasepool {
      return ((id<MTLTexture>)texture).tailSizeInBytes;
   }
}

void
mtl_command_queue_update_buffer_mappings(
   mtl_command_queue *queue, mtl_buffer *buffer, mtl_heap *heap,
   const struct mtl_sparse_buffer_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      MTL4UpdateSparseBufferMappingOperation *ops =
         malloc(count * sizeof(*ops));
      if (!ops)
         return;
      for (uint32_t i = 0; i < count; i++) {
         ops[i] = (MTL4UpdateSparseBufferMappingOperation){
            .mode = mappings[i].map ? MTLSparseTextureMappingModeMap
                                    : MTLSparseTextureMappingModeUnmap,
            .bufferRange = NSMakeRange(mappings[i].first_page, mappings[i].page_count),
            .heapOffset = mappings[i].heap_page,
         };
      }
      [(id<MTL4CommandQueue>)queue updateBufferMappings:(id<MTLBuffer>)buffer
                                                   heap:(id<MTLHeap>)heap
                                             operations:ops
                                                  count:count];
      free(ops);
   }
}

void
mtl_command_queue_update_texture_mappings(
   mtl_command_queue *queue, mtl_texture *texture, mtl_heap *heap,
   const struct mtl_sparse_texture_mapping *mappings, uint32_t count)
{
   @autoreleasepool {
      MTL4UpdateSparseTextureMappingOperation *ops =
         malloc(count * sizeof(*ops));
      if (!ops)
         return;
      for (uint32_t i = 0; i < count; i++) {
         const struct mtl_sparse_texture_mapping *m = &mappings[i];
         ops[i] = (MTL4UpdateSparseTextureMappingOperation){
            .mode = m->map ? MTLSparseTextureMappingModeMap
                           : MTLSparseTextureMappingModeUnmap,
            .textureRegion = MTLRegionMake3D(m->origin.x, m->origin.y, m->origin.z,
                                             m->size.x, m->size.y, m->size.z),
            .textureLevel = m->level,
            .textureSlice = m->slice,
            .heapOffset = m->heap_page,
         };
      }
      [(id<MTL4CommandQueue>)queue updateTextureMappings:(id<MTLTexture>)texture
                                                    heap:(id<MTLHeap>)heap
                                              operations:ops
                                                   count:count];
      free(ops);
   }
}
