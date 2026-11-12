/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_rt_pipeline.h"

#include <Metal/MTL4BinaryFunction.h>
#include <Metal/MTL4BinaryFunctionDescriptor.h>
#include <Metal/MTL4Compiler.h>
#include <Metal/MTL4ComputePipeline.h>
#include <Metal/MTL4LibraryFunctionDescriptor.h>
#include <Metal/MTL4LinkingDescriptor.h>
#include <Metal/MTLComputePipeline.h>
#include <Metal/MTLVisibleFunctionTable.h>

mtl_binary_function *
mtl_new_binary_function(mtl_compiler *compiler, mtl_library *library,
                        const char *name)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      NSString *ns_name = [NSString stringWithCString:name encoding:NSASCIIStringEncoding];

      MTL4LibraryFunctionDescriptor *function = [[MTL4LibraryFunctionDescriptor new] autorelease];
      function.library = (id<MTLLibrary>)library;
      function.name = ns_name;

      MTL4BinaryFunctionDescriptor *desc = [[MTL4BinaryFunctionDescriptor new] autorelease];
      desc.name = ns_name;
      desc.functionDescriptor = function;
      desc.options = MTL4BinaryFunctionOptionPipelineIndependent;

      NSError *error;
      id<MTL4BinaryFunction> result = [comp newBinaryFunctionWithDescriptor:desc
                                                         compilerTaskOptions:nil
                                                                       error:&error];
      if (result == nil)
         fprintf(stderr, "Failed to create MTL4BinaryFunction %s: %s\n", name,
                 [error.localizedDescription UTF8String]);
      return result;
   }
}

mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state_with_functions(
   mtl_compiler *compiler, mtl_function_descriptor *kernel,
   mtl_binary_function *const *functions, uint32_t function_count,
   uint32_t max_call_stack_depth)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;

      MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)kernel;
      desc.supportBinaryLinking = YES;

      NSMutableArray<id<MTL4BinaryFunction>> *linked =
         [NSMutableArray arrayWithCapacity:function_count];
      for (uint32_t i = 0; i < function_count; i++)
         [linked addObject:(id<MTL4BinaryFunction>)functions[i]];

      MTL4PipelineStageDynamicLinkingDescriptor *linking =
         [[MTL4PipelineStageDynamicLinkingDescriptor new] autorelease];
      linking.binaryLinkedFunctions = linked;
      linking.maxCallStackDepth = max_call_stack_depth;

      NSError *error;
      id<MTLComputePipelineState> pipeline =
         [comp newComputePipelineStateWithDescriptor:desc
                            dynamicLinkingDescriptor:linking
                                 compilerTaskOptions:nil
                                               error:&error];
      if (pipeline == nil)
         fprintf(stderr, "Failed to link ray tracing pipeline: %s\n",
                 [error.localizedDescription UTF8String]);
      return pipeline;
   }
}

uint64_t
mtl_compute_pipeline_function_id(mtl_compute_pipeline_state *pipeline,
                                 mtl_binary_function *function)
{
   @autoreleasepool {
      id<MTLComputePipelineState> pipe = (id<MTLComputePipelineState>)pipeline;
      id<MTLFunctionHandle> handle =
         [pipe functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function];
      return handle ? handle.gpuResourceID._impl : 0u;
   }
}

mtl_visible_function_table *
mtl_new_visible_function_table(mtl_compute_pipeline_state *pipeline,
                               uint32_t function_count)
{
   @autoreleasepool {
      id<MTLComputePipelineState> pipe = (id<MTLComputePipelineState>)pipeline;
      MTLVisibleFunctionTableDescriptor *desc =
         [[MTLVisibleFunctionTableDescriptor new] autorelease];
      desc.functionCount = function_count;
      return [pipe newVisibleFunctionTableWithDescriptor:desc];
   }
}

void
mtl_visible_function_table_set(mtl_visible_function_table *table,
                               mtl_compute_pipeline_state *pipeline,
                               uint32_t index, mtl_binary_function *function)
{
   @autoreleasepool {
      id<MTLComputePipelineState> pipe = (id<MTLComputePipelineState>)pipeline;
      id<MTLVisibleFunctionTable> vft = (id<MTLVisibleFunctionTable>)table;
      [vft setFunction:[pipe functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function]
               atIndex:index];
   }
}

uint64_t
mtl_visible_function_table_resource_id(mtl_visible_function_table *table)
{
   @autoreleasepool {
      id<MTLVisibleFunctionTable> vft = (id<MTLVisibleFunctionTable>)table;
      return vft.gpuResourceID._impl;
   }
}
