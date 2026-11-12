/*
 * SPDX-License-Identifier: MIT
 */

#include "mtl_rt_pipeline.h"

mtl_binary_function *
mtl_new_binary_function(mtl_compiler *compiler, mtl_library *library,
                        const char *name)
{
   return NULL;
}

mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state_with_functions(
   mtl_compiler *compiler, mtl_function_descriptor *kernel,
   mtl_binary_function *const *functions, uint32_t function_count,
   uint32_t max_call_stack_depth)
{
   return NULL;
}

uint64_t
mtl_compute_pipeline_function_id(mtl_compute_pipeline_state *pipeline,
                                 mtl_binary_function *function)
{
   return 0u;
}

mtl_visible_function_table *
mtl_new_visible_function_table(mtl_compute_pipeline_state *pipeline,
                               uint32_t function_count)
{
   return NULL;
}

void
mtl_visible_function_table_set(mtl_visible_function_table *table,
                               mtl_compute_pipeline_state *pipeline,
                               uint32_t index, mtl_binary_function *function)
{
}

uint64_t
mtl_visible_function_table_resource_id(mtl_visible_function_table *table)
{
   return 0u;
}
