/*
 * SPDX-License-Identifier: MIT
 */

#include "nir_to_msl.h"

#include "nir_builder.h"
#include "util/hash_table.h"

/* Ray queries become Metal intersection_query objects, declared by nir_to_msl
 * as one array, rq[shader->info.ray_queries]. This pass gives every ray query
 * variable (and every element of a ray query array) an index into it, and
 * rewrites the rq_* intrinsics to take that index instead of a deref. It runs
 * right after inlining, before anything else touches the opaque variables. */

static nir_def *
ray_query_index(nir_builder *b, nir_deref_instr *deref, struct hash_table *bases)
{
   if (deref->deref_type == nir_deref_type_var) {
      struct hash_entry *entry = _mesa_hash_table_search(bases, deref->var);
      assert(entry);
      return nir_imm_int(b, (uint32_t)(uintptr_t)entry->data);
   }

   assert(deref->deref_type == nir_deref_type_array);
   nir_def *parent = ray_query_index(b, nir_deref_instr_parent(deref), bases);
   const unsigned stride = glsl_type_is_array(deref->type)
                              ? glsl_get_aoa_size(deref->type)
                              : 1u;
   nir_def *element = nir_u2u32(b, deref->arr.index.ssa);
   return nir_iadd(b, parent, nir_imul_imm(b, element, stride));
}

static void
assign_bases(struct exec_list *vars, struct hash_table *bases,
             unsigned *count)
{
   nir_foreach_variable_in_list(var, vars) {
      if (!var->data.ray_query)
         continue;

      _mesa_hash_table_insert(bases, var, (void *)(uintptr_t)*count);
      *count += glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type)
                                              : 1u;
   }
}

bool
msl_nir_lower_ray_queries(nir_shader *nir)
{
   struct hash_table *bases = _mesa_pointer_hash_table_create(NULL);
   unsigned count = 0;

   assign_bases(&nir->variables, bases, &count);
   nir_foreach_function_impl(impl, nir)
      assign_bases(&impl->locals, bases, &count);

   if (count == 0) {
      _mesa_hash_table_destroy(bases, NULL);
      return false;
   }

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_rq_initialize:
            case nir_intrinsic_rq_terminate:
            case nir_intrinsic_rq_proceed:
            case nir_intrinsic_rq_generate_intersection:
            case nir_intrinsic_rq_confirm_intersection:
            case nir_intrinsic_rq_load:
               break;
            default:
               continue;
            }

            b.cursor = nir_before_instr(instr);
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            nir_src_rewrite(&intrin->src[0],
                            ray_query_index(&b, deref, bases));
         }
      }

      nir_progress(true, impl, nir_metadata_control_flow);
   }

   _mesa_hash_table_destroy(bases, NULL);

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_function_temp | nir_var_shader_temp, NULL);
   nir->info.ray_queries = count;

   /* nir_shader_gather_info recounts info.ray_queries from ray query
    * variables (typed uint64 by spirv_to_nir), so one unreferenced variable
    * stays to stand for rq[]. Only
    * function temporaries and I/O are removed when dead after this point. */
   nir_variable *placeholder = nir_variable_create(
      nir, nir_var_shader_temp,
      glsl_array_type(glsl_uint64_t_type(), count, 0), "rq");
   placeholder->data.ray_query = true;

   return true;
}
