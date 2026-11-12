/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_acceleration_structure.h"

#include <Metal/MTL4AccelerationStructure.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTLAccelerationStructure.h>
#include <Metal/MTLDevice.h>

static_assert(sizeof(struct mtl_indirect_instance_descriptor) ==
                 sizeof(MTLIndirectAccelerationStructureInstanceDescriptor),
              "instance descriptor layout must match Metal's");

static MTL4BufferRange
mtl_range(uint64_t addr, uint64_t size_B)
{
   return (MTL4BufferRange){
      .bufferAddress = addr,
      .length = size_B ? size_B : (uint64_t)-1,
   };
}

static MTL4AccelerationStructureGeometryDescriptor *
mtl_new_geometry_descriptor(const struct mtl_acceleration_structure_geometry *geo)
{
   MTL4AccelerationStructureGeometryDescriptor *result;

   if (geo->kind == MTL_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES) {
      MTL4AccelerationStructureTriangleGeometryDescriptor *tri =
         [[MTL4AccelerationStructureTriangleGeometryDescriptor new] autorelease];
      tri.vertexBuffer = mtl_range(geo->vertex_addr, geo->vertex_size_B);
      tri.vertexStride = geo->vertex_stride_B;
      tri.vertexFormat = (MTLAttributeFormat)geo->vertex_format;
      if (geo->index_addr) {
         tri.indexBuffer = mtl_range(
            geo->index_addr,
            (uint64_t)geo->primitive_count * 3u * geo->index_size_B);
         tri.indexType =
            geo->index_size_B == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
      }
      if (geo->transform_addr) {
         /* VkTransformMatrixKHR is a row-major 3x4 matrix */
         tri.transformationMatrixBuffer =
            mtl_range(geo->transform_addr, 12u * sizeof(float));
         tri.transformationMatrixLayout = MTLMatrixLayoutRowMajor;
      }
      tri.triangleCount = geo->primitive_count;
      result = tri;
   } else {
      MTL4AccelerationStructureBoundingBoxGeometryDescriptor *box =
         [[MTL4AccelerationStructureBoundingBoxGeometryDescriptor new]
            autorelease];
      box.boundingBoxBuffer =
         mtl_range(geo->aabb_addr,
                   (uint64_t)geo->primitive_count * geo->aabb_stride_B);
      box.boundingBoxStride = geo->aabb_stride_B;
      box.boundingBoxCount = geo->primitive_count;
      result = box;
   }

   result.opaque = geo->opaque;
   result.allowDuplicateIntersectionFunctionInvocation =
      geo->allow_duplicate_any_hit;
   /* Vulkan's geometry index reaches the shader binding table through the
    * intersector's geometry multiplier, so no per-geometry offset here. */
   result.intersectionFunctionTableOffset = 0;
   return result;
}

static MTL4AccelerationStructureDescriptor *
mtl_new_descriptor(const struct mtl_acceleration_structure_desc *desc)
{
   MTL4AccelerationStructureDescriptor *result;

   if (desc->is_instance) {
      MTL4InstanceAccelerationStructureDescriptor *inst =
         [[MTL4InstanceAccelerationStructureDescriptor new] autorelease];
      inst.instanceDescriptorBuffer =
         mtl_range(desc->instance_addr,
                   (uint64_t)desc->instance_count *
                      sizeof(MTLIndirectAccelerationStructureInstanceDescriptor));
      inst.instanceDescriptorStride =
         sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
      inst.instanceCount = desc->instance_count;
      inst.instanceDescriptorType =
         MTLAccelerationStructureInstanceDescriptorTypeIndirect;
      /* Vulkan instance transforms are row-major 3x4, copied as they are */
      inst.instanceTransformationMatrixLayout = MTLMatrixLayoutRowMajor;
      result = inst;
   } else {
      NSMutableArray<MTL4AccelerationStructureGeometryDescriptor *> *geometries =
         [NSMutableArray arrayWithCapacity:desc->geometry_count];
      for (uint32_t i = 0; i < desc->geometry_count; i++)
         [geometries addObject:mtl_new_geometry_descriptor(&desc->geometries[i])];

      MTL4PrimitiveAccelerationStructureDescriptor *prim =
         [[MTL4PrimitiveAccelerationStructureDescriptor new] autorelease];
      prim.geometryDescriptors = geometries;
      result = prim;
   }

   result.usage = (MTLAccelerationStructureUsage)desc->usage;
   return result;
}

void
mtl_acceleration_structure_get_sizes(
   mtl_device *device, const struct mtl_acceleration_structure_desc *desc,
   struct mtl_acceleration_structure_sizes *sizes)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLAccelerationStructureSizes s =
         [dev accelerationStructureSizesWithDescriptor:mtl_new_descriptor(desc)];
      sizes->acceleration_structure_size_B = s.accelerationStructureSize;
      sizes->build_scratch_size_B = s.buildScratchBufferSize;
      sizes->refit_scratch_size_B = s.refitScratchBufferSize;
   }
}


mtl_acceleration_structure *
mtl_new_acceleration_structure(mtl_device *device, uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newAccelerationStructureWithSize:size_B];
   }
}


uint64_t
mtl_acceleration_structure_get_resource_id(mtl_acceleration_structure *as)
{
   @autoreleasepool {
      id<MTLAccelerationStructure> accel = (id<MTLAccelerationStructure>)as;
      return accel.gpuResourceID._impl;
   }
}

void
mtl_build_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *as,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc buildAccelerationStructure:(id<MTLAccelerationStructure>)as
                           descriptor:mtl_new_descriptor(desc)
                        scratchBuffer:mtl_range(scratch_addr, scratch_size_B)];
   }
}

void
mtl_refit_acceleration_structure(
   mtl_compute_encoder *encoder, mtl_acceleration_structure *src,
   mtl_acceleration_structure *dst,
   const struct mtl_acceleration_structure_desc *desc, uint64_t scratch_addr,
   uint64_t scratch_size_B)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc refitAccelerationStructure:(id<MTLAccelerationStructure>)src
                           descriptor:mtl_new_descriptor(desc)
                          destination:(id<MTLAccelerationStructure>)dst
                        scratchBuffer:mtl_range(scratch_addr, scratch_size_B)
                              options:MTLAccelerationStructureRefitOptionVertexData |
                                      MTLAccelerationStructureRefitOptionPerPrimitiveData];
   }
}

void
mtl_copy_acceleration_structure(mtl_compute_encoder *encoder,
                                mtl_acceleration_structure *src,
                                mtl_acceleration_structure *dst)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc copyAccelerationStructure:(id<MTLAccelerationStructure>)src
             toAccelerationStructure:(id<MTLAccelerationStructure>)dst];
   }
}

void
mtl_copy_and_compact_acceleration_structure(mtl_compute_encoder *encoder,
                                            mtl_acceleration_structure *src,
                                            mtl_acceleration_structure *dst)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc copyAndCompactAccelerationStructure:(id<MTLAccelerationStructure>)src
                       toAccelerationStructure:(id<MTLAccelerationStructure>)dst];
   }
}

void
mtl_write_compacted_acceleration_structure_size(mtl_compute_encoder *encoder,
                                                mtl_acceleration_structure *as,
                                                uint64_t addr)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      [enc writeCompactedAccelerationStructureSize:(id<MTLAccelerationStructure>)as
                                          toBuffer:mtl_range(addr, sizeof(uint64_t))];
   }
}
