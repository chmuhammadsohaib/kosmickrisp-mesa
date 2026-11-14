/*
 * SPDX-License-Identifier: MIT
 */
#include "kk_dispatch_trace.h"

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/u_atomic.h"
#include "util/u_debug.h"

#include <inttypes.h>
#include <stdio.h>

/* Mode 0 starts a record (copying 3 group counts from src when it isn't 0),
 * mode 1 finishes it */
static const char kk_dispatch_trace_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "struct Trace { ulong record, src; uint mode; };\n"
   "struct Record { uint seq, indirect, groups[3], local[3]; atomic_uint started, done; };\n"
   "kernel void kk_dispatch_trace(constant Trace &t [[buffer(0)]])\n"
   "{\n"
   "   device Record *r = (device Record *)t.record;\n"
   "   if (t.mode == 1u) {\n"
   "      atomic_store_explicit(&r->done, 1u, memory_order_relaxed);\n"
   "      return;\n"
   "   }\n"
   "   if (t.src != 0ul) {\n"
   "      device const uint *g = (device const uint *)t.src;\n"
   "      r->groups[0] = g[0];\n"
   "      r->groups[1] = g[1];\n"
   "      r->groups[2] = g[2];\n"
   "   }\n"
   "   atomic_store_explicit(&r->started, 1u, memory_order_relaxed);\n"
   "}\n";

struct kk_dispatch_trace_params {
   uint64_t record, src;
   uint32_t mode;
};

void
kk_dispatch_trace_init(struct kk_device *dev)
{
   if (!debug_get_bool_option("KK_DISPATCH_DEBUG", false))
      return;

   const size_t size = KK_DISPATCH_TRACE_RECORDS * sizeof(struct kk_dispatch_record);
   if (kk_alloc_bo(dev, &dev->vk.base, size, 0, &dev->dispatch_trace.bo) !=
       VK_SUCCESS)
      return;
   memset(dev->dispatch_trace.bo->cpu, 0, size);

   mtl_library *lib = mtl_new_library(
      dev->mtl_compiler_handle, kk_dispatch_trace_msl, MTL_LANGUAGE_VERSION_4_0,
      MTL_MATH_MODE_FAST, MTL_MATH_FLOATING_POINT_FUNCTIONS_FAST);
   mtl_function_descriptor *fn =
      lib ? mtl_new_library_function_descriptor(lib, "kk_dispatch_trace") : NULL;
   if (fn)
      dev->dispatch_trace.pipeline =
         mtl_new_compute_pipeline_state(dev->mtl_compiler_handle, fn, 1);
   if (fn)
      mtl_release(fn);
   if (lib)
      mtl_release(lib);
   if (!dev->dispatch_trace.pipeline) {
      kk_destroy_bo(dev, dev->dispatch_trace.bo);
      dev->dispatch_trace.bo = NULL;
   }
}

void
kk_dispatch_trace_finish(struct kk_device *dev)
{
   kk_dispatch_trace_report(dev);
   if (dev->dispatch_trace.pipeline)
      mtl_release(dev->dispatch_trace.pipeline);
   if (dev->dispatch_trace.bo)
      kk_destroy_bo(dev, dev->dispatch_trace.bo);
}

static void
kk_dispatch_trace_run(struct kk_cmd_buffer *cmd, uint64_t record, uint64_t src,
                      uint32_t mode)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_ptr p = kk_pool_alloc(cmd, sizeof(struct kk_dispatch_trace_params), 8u);
   if (!p.gpu)
      return;
   *(struct kk_dispatch_trace_params *)p.cpu =
      (struct kk_dispatch_trace_params){record, src, mode};
   mtl_compute_encoder *enc = cs_get_compute(cmd);
   mtl_set_address(cmd->argument_table, p.gpu, 0u);
   mtl_compute_set_pipeline_state(enc, dev->dispatch_trace.pipeline);
   mtl_dispatch_threads(enc, (struct mtl_size){1, 1, 1},
                        (struct mtl_size){1, 1, 1});
}

uint64_t
kk_dispatch_trace_begin(struct kk_cmd_buffer *cmd, const struct kk_shader *cs,
                        const uint32_t groups[3], uint64_t indirect_addr)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   if (likely(!dev->dispatch_trace.bo))
      return 0u;

   uint32_t seq = p_atomic_inc_return(&dev->dispatch_trace.seq);
   uint32_t i = (seq - 1u) % KK_DISPATCH_TRACE_RECORDS;
   struct kk_dispatch_record *r =
      (struct kk_dispatch_record *)dev->dispatch_trace.bo->cpu + i;
   *r = (struct kk_dispatch_record){
      .seq = seq,
      .indirect = indirect_addr != 0u,
      .groups = {groups[0], groups[1], groups[2]},
      .local = {cs->info.cs.local_size.x, cs->info.cs.local_size.y,
                cs->info.cs.local_size.z},
      .shader = (uintptr_t)cs,
      .indirect_addr = indirect_addr,
   };
   uint64_t record = dev->dispatch_trace.bo->gpu + i * sizeof(*r);

   mtl_compute_encoder *enc = cs_get_compute(cmd);
   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH, MTL_STAGE_DISPATCH);
   kk_dispatch_trace_run(cmd, record, indirect_addr, 0u);
   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH, MTL_STAGE_DISPATCH);
   return record;
}

void
kk_dispatch_trace_end(struct kk_cmd_buffer *cmd, uint64_t record)
{
   if (likely(!record))
      return;
   mtl_compute_encoder *enc = cs_get_compute(cmd);
   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH, MTL_STAGE_DISPATCH);
   kk_dispatch_trace_run(cmd, record, 0u, 1u);
}

void
kk_dispatch_trace_report(struct kk_device *dev)
{
   if (!dev->dispatch_trace.bo)
      return;
   const struct kk_dispatch_record *r = dev->dispatch_trace.bo->cpu;
   uint32_t last = p_atomic_read(&dev->dispatch_trace.seq);
   uint32_t first = last > KK_DISPATCH_TRACE_RECORDS
                       ? last - KK_DISPATCH_TRACE_RECORDS + 1u
                       : 1u;
   uint32_t running = 0u, pending = 0u;
   for (uint32_t seq = first; seq <= last; seq++) {
      const struct kk_dispatch_record *e = &r[(seq - 1u) % KK_DISPATCH_TRACE_RECORDS];
      if (e->seq != seq || e->done)
         continue;
      if (!e->started) {
         pending++;
         continue;
      }
      if (running++ < 16u)
         fprintf(stderr,
                 "[kk] dispatch debug: dispatch %u started, not done: %s %u x %u x %u "
                 "groups of %u x %u x %u, shader 0x%" PRIx64 "\n",
                 seq, e->indirect ? "indirect" : "direct", e->groups[0],
                 e->groups[1], e->groups[2], e->local[0], e->local[1],
                 e->local[2], e->shader);
   }
   fprintf(stderr, "[kk] dispatch debug: %u dispatches recorded, %u started but "
           "not done, %u not started\n", last, running, pending);
}
