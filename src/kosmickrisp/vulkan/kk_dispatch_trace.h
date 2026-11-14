/*
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_DISPATCH_TRACE_H
#define KK_DISPATCH_TRACE_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include <stdint.h>

struct kk_bo;
struct kk_device;
struct kk_cmd_buffer;
struct kk_shader;

/* KK_DISPATCH_DEBUG=1: a record per compute dispatch in a ring the CPU reads
 * when the device is lost. A one-thread dispatch before it marks it started
 * and, for an indirect dispatch, copies the group counts the GPU will use; one
 * after it (behind a barrier) marks it done. A dispatch started but not done
 * is the one the GPU was running, with the size it ran at. Every dispatch is
 * serialized while this is on. */
#define KK_DISPATCH_TRACE_RECORDS 16384u

struct kk_dispatch_record {
   uint32_t seq;               /* dispatch number, from 1 */
   uint32_t indirect;
   uint32_t groups[3];         /* GPU-copied for an indirect dispatch */
   uint32_t local[3];
   uint32_t started, done;     /* GPU */
   uint64_t shader;            /* struct kk_shader *, to tell shaders apart */
   uint64_t indirect_addr;
};

struct kk_dispatch_trace {
   mtl_compute_pipeline_state *pipeline;
   struct kk_bo *bo;
   uint32_t seq;
};

void kk_dispatch_trace_init(struct kk_device *dev);
void kk_dispatch_trace_finish(struct kk_device *dev);

/* Around a dispatch: begin returns the record's address (0 when off), end
 * marks it done. Both leave the compute pipeline and root unbound: the next
 * dispatch's state flush binds them again. */
uint64_t kk_dispatch_trace_begin(struct kk_cmd_buffer *cmd,
                                 const struct kk_shader *cs,
                                 const uint32_t groups[3],
                                 uint64_t indirect_addr);
void kk_dispatch_trace_end(struct kk_cmd_buffer *cmd, uint64_t record);

/* Prints the dispatches started but not done */
void kk_dispatch_trace_report(struct kk_device *dev);

#endif
