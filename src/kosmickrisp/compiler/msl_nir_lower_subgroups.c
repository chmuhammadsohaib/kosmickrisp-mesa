/*
 * Copyright 2023 Valve Corporation
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "msl_private.h"
#include "nir.h"
#include "nir_builder.h"

#include "util/u_debug.h"

static bool
needs_bool_widening(nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_reduce:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_rotate:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_xor:
      return true;
   default:
      return false;
   }
}

static bool
lower_bool_ops(nir_builder *b, nir_intrinsic_instr *intrin, void *_unused)
{
   if (!needs_bool_widening(intrin))
      return false;

   if (intrin->def.bit_size != 1)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *widen = nir_b2i32(b, intrin->src[0].ssa);
   nir_src_rewrite(&intrin->src[0], widen);
   intrin->def.bit_size = 32;
   b->cursor = nir_after_instr(&intrin->instr);
   nir_def *narrow = nir_b2b1(b, &intrin->def);
   nir_def_rewrite_uses_after(&intrin->def, narrow);

   return true;
}

/* Which subgroup operations nir_lower_subgroups rewrites. Reductions over the
 * whole subgroup of 32-bit integers become one Metal simd_* instruction instead
 * of a loop of shuffles; float sums and products too, unless KK_EXACT=1: Metal
 * adds in its own order, so the last bits can differ from the lowered order.
 * Float min/max stay lowered, which keeps NIR's NaN rules. */
static bool
lower_subgroups_filter(const nir_intrinsic_instr *intr, const void *data)
{
   if (intr->intrinsic != nir_intrinsic_reduce)
      return true;
   const unsigned cluster = nir_intrinsic_cluster_size(intr);
   if ((cluster != 0 && cluster != 32) || intr->def.bit_size != 32)
      return true;

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_iadd:
   case nir_op_imul:
   case nir_op_imin:
   case nir_op_umin:
   case nir_op_imax:
   case nir_op_umax:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      return false;
   case nir_op_fadd:
   case nir_op_fmul:
      return debug_get_bool_option("KK_EXACT", false);
   default:
      return true;
   }
}

void
msl_nir_lower_subgroups(nir_shader *nir)
{
   const nir_lower_subgroups_options subgroups_options = {
      .subgroup_size = 32,
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_subgroup_masks = true,
      .lower_vote_ieq = true,
      .lower_vote_feq = true,
      .lower_vote_bool_eq = true,
      .lower_inverse_ballot = true,
      /* Metal requires relative shuffle operations to have uniform delta */
      .lower_relative_shuffle = true,
      /* Metal reduce operations do not support certain types or cluster size */
      .lower_reduce = true,
      /* Metal rotate operations do not support cluster size */
      .lower_rotate_clustered_to_shuffle = true,
      .filter = lower_subgroups_filter,
   };
   NIR_PASS(_, nir, nir_lower_subgroups, &subgroups_options);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_bool_ops,
            nir_metadata_control_flow, NULL);
}
