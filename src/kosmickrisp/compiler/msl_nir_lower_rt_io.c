/*
 * SPDX-License-Identifier: MIT
 */

#include "nir_to_msl.h"

#include "nir_builder.h"
#include "util/hash_table.h"

/* Ray tracing stages keep payloads, callable data and hit attributes in device
 * memory (see msl_rt.c): the incoming ones are in the stage's kk_rt_call
 * frame, the outgoing ones in an area right after it. Shader record buffers
 * are global memory at the record's address. This pass turns every
 * deref of those variables into a cast from the matching address. Global
 * memory lowering does the rest, turning the payload derefs trace_ray and
 * execute_callable take into the addresses nir_to_msl passes on. */

static const struct glsl_type *
explicit_type(const struct glsl_type *type, unsigned *size_B, unsigned *align_B)
{
   return glsl_get_explicit_type_for_size_align(
      type, glsl_get_natural_size_align_bytes, size_B, align_B);
}

static nir_variable *
call_payload_var(nir_intrinsic_instr *intrin)
{
   unsigned src;
   switch (intrin->intrinsic) {
   case nir_intrinsic_trace_ray:
      src = 10;
      break;
   case nir_intrinsic_execute_callable:
      src = 1;
      break;
   default:
      return NULL;
   }
   return nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[src]));
}

/* Gives each outgoing payload and callable data variable an offset in the
 * outgoing area. Returns the area's size. */
static unsigned
assign_out_offsets(nir_shader *nir, struct hash_table *offsets)
{
   unsigned size_B = 0;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_variable *var = call_payload_var(nir_instr_as_intrinsic(instr));
            if (!var || _mesa_hash_table_search(offsets, var))
               continue;

            unsigned var_size_B, var_align_B;
            explicit_type(var->type, &var_size_B, &var_align_B);
            size_B = align(size_B, MAX2(var_align_B, 1u));
            _mesa_hash_table_insert(offsets, var, (void *)(uintptr_t)size_B);
            size_B += var_size_B;
         }
      }
   }

   return align(size_B, 16u);
}

static nir_def *
var_address(nir_builder *b, nir_variable *var, struct hash_table *offsets,
            unsigned out_size_B)
{
   struct hash_entry *entry = _mesa_hash_table_search(offsets, var);
   if (entry) {
      return nir_iadd_imm(b, nir_load_rt_out_ptr_kk(b, .range = out_size_B),
                          (uintptr_t)entry->data);
   }

   switch (var->data.mode) {
   case nir_var_shader_call_data:
      return nir_load_rt_payload_ptr_kk(b);
   case nir_var_ray_hit_attrib:
      return nir_load_rt_attrib_ptr_kk(b);
   default:
      return NULL;
   }
}

bool
msl_nir_lower_rt_io(nir_shader *nir)
{
   if (!mesa_shader_stage_is_rt(nir->info.stage))
      return false;

   struct hash_table *offsets = _mesa_pointer_hash_table_create(NULL);
   const unsigned out_size_B = assign_out_offsets(nir, offsets);
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);

            /* spirv_to_nir casts the shader record pointer as constant
             * memory, which is only lowered for real constants; it is
             * global memory here. */
            if (deref->deref_type == nir_deref_type_cast &&
                deref->modes == nir_var_mem_constant) {
               nir_intrinsic_instr *ptr = nir_src_as_intrinsic(deref->parent);
               if (ptr &&
                   ptr->intrinsic == nir_intrinsic_load_shader_record_ptr) {
                  deref->modes = nir_var_mem_global;
                  progress = true;
               }
               continue;
            }

            if (deref->deref_type != nir_deref_type_var)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *addr = var_address(&b, deref->var, offsets, out_size_B);
            if (!addr)
               continue;

            unsigned size_B, align_B;
            const struct glsl_type *type =
               explicit_type(deref->var->type, &size_B, &align_B);
            nir_deref_instr *cast =
               nir_build_deref_cast(&b, addr, nir_var_mem_global, type, 0);
            cast->cast.align_mul = MAX2(align_B, 1u);
            nir_def_rewrite_uses(&deref->def, &cast->def);
            progress = true;
         }
      }

      nir_progress(progress, impl, nir_metadata_control_flow);
   }

   _mesa_hash_table_destroy(offsets, NULL);

   if (progress) {
      /* Function temporaries had 32-bit derefs; the casts are 64-bit global
       * addresses, so their children must follow. Parents dominate their
       * children, so one pass in block order sees parents first. */
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_deref)
                  continue;
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (deref->deref_type == nir_deref_type_var ||
                   deref->deref_type == nir_deref_type_cast)
                  continue;
               nir_deref_instr *parent = nir_deref_instr_parent(deref);
               deref->def.bit_size = parent->def.bit_size;
               deref->def.num_components = parent->def.num_components;
            }
         }
      }
      nir_fixup_deref_types(nir);
      nir_fixup_deref_modes(nir);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_derefs);
      NIR_PASS(_, nir, nir_remove_dead_variables,
               nir_var_shader_call_data | nir_var_ray_hit_attrib |
                  nir_var_shader_temp | nir_var_function_temp,
               NULL);
   }
   return progress;
}

/* After global memory lowering, the payloads trace_ray and execute_callable
 * take are addresses. Switch to the KosmicKrisp versions, which other passes
 * don't expect a deref from. */
static bool
lower_rt_call(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   nir_intrinsic_op op;
   switch (intrin->intrinsic) {
   case nir_intrinsic_trace_ray:
      op = nir_intrinsic_trace_ray_kk;
      break;
   case nir_intrinsic_execute_callable:
      op = nir_intrinsic_execute_callable_kk;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intrin->instr);
   nir_intrinsic_instr *call = nir_intrinsic_instr_create(b->shader, op);
   const unsigned num_srcs = nir_intrinsic_infos[op].num_srcs;
   for (unsigned i = 0; i < num_srcs; i++)
      call->src[i] = nir_src_for_ssa(intrin->src[i].ssa);
   nir_builder_instr_insert(b, &call->instr);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
msl_nir_lower_rt_calls(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_rt_call,
                                     nir_metadata_control_flow, NULL);
}
