/*
 * SPDX-License-Identifier: MIT
 */

#ifndef MSL_RT_H
#define MSL_RT_H

#include <stdint.h>

/* Layout of kk_rt_call in msl_rt_preamble, for the driver */
#define MSL_RT_FRAME_SIZE        512u
#define MSL_RT_RECORD_FN_OFFSET  8u
#define MSL_RT_HANDLE_SIZE       32u

/* nir_to_msl names a ray tracing stage this; the driver replaces it with a
 * name unique to the shader, as all stages are linked into one pipeline. */
#define MSL_RT_ENTRY_PLACEHOLDER "kk_rt_entry"

/* any-hit outcomes in kk_rt_call::ahit_status */
#define MSL_RT_AHIT_ACCEPT    0u
#define MSL_RT_AHIT_IGNORE    1u
#define MSL_RT_AHIT_TERMINATE 2u

/* kk_rt_dispatch_args in msl_rt_dispatch_source */
struct msl_rt_dispatch_args {
   uint64_t root;
   uint64_t sampler_table;
   uint64_t vft;
   uint64_t hit_sbt;
   uint64_t miss_sbt;
   uint64_t callable_sbt;
   uint64_t hit_sbt_size;
   uint64_t raygen_record;
   uint64_t stack_base;
   uint32_t hit_stride;
   uint32_t miss_stride;
   uint32_t callable_stride;
   uint32_t stack_size;
   uint32_t launch_size[3];
   uint32_t pad;
   uint64_t indirect;
   uint64_t vft_id;
};

extern const char *msl_rt_preamble;
extern const char *msl_rt_dispatch_source;

#endif /* MSL_RT_H */
