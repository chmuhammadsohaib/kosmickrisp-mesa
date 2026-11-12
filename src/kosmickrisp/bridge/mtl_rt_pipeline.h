/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_RT_PIPELINE_H
#define MTL_RT_PIPELINE_H 1

#include "mtl_types.h"

#include <inttypes.h>

/* A pipeline-independent binary function, whose function handle has the same
 * resource ID in every pipeline that links it. */
mtl_binary_function *mtl_new_binary_function(mtl_compiler *compiler,
                                             mtl_library *library,
                                             const char *name);

/* A compute pipeline for kernel that links functions, which its visible
 * function tables and intersection function buffers can then reference. */
mtl_compute_pipeline_state *mtl_new_compute_pipeline_state_with_functions(
   mtl_compiler *compiler, mtl_function_descriptor *kernel,
   mtl_binary_function *const *functions, uint32_t function_count,
   uint32_t max_call_stack_depth);

/* Resource ID of a linked function's handle */
uint64_t mtl_compute_pipeline_function_id(mtl_compute_pipeline_state *pipeline,
                                          mtl_binary_function *function);

mtl_visible_function_table *
mtl_new_visible_function_table(mtl_compute_pipeline_state *pipeline,
                               uint32_t function_count);

void mtl_visible_function_table_set(mtl_visible_function_table *table,
                                    mtl_compute_pipeline_state *pipeline,
                                    uint32_t index,
                                    mtl_binary_function *function);

uint64_t
mtl_visible_function_table_resource_id(mtl_visible_function_table *table);

#endif /* MTL_RT_PIPELINE_H */
