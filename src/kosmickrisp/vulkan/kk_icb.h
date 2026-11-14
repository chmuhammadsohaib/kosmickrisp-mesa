/*
 * SPDX-License-Identifier: MIT
 */
#ifndef KK_ICB_H
#define KK_ICB_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

#include <stdbool.h>
#include <stdint.h>

struct kk_device;
struct kk_cmd_buffer;

/* KK_ICB_DRAWS (off by default): vkCmdDraw*IndirectCount as GPU-built draws.
 * A rasterization-off vertex pass in the render pass writes the draws the
 * count buffer asks for into an indirect command buffer, then the ICB runs
 * with that count as its range: no Metal draw per possible draw, no compute
 * pass that splits the render pass. */

/* Slots in a pooled ICB; calls with more possible draws get one of their own */
#define KK_ICB_CHUNK_SLOTS 4096u

struct kk_icb_chunk {
   mtl_indirect_command_buffer *handle;
   uint64_t resource_id;
   uint32_t capacity;
};

struct kk_icb_device {
   mtl_render_pipeline_state *encode_pipeline;
   simple_mtx_t lock;
   struct util_dynarray free_chunks; /* struct kk_icb_chunk */
};

/* Per command buffer: the chunks its draws use, until it's reset. Calls reuse
 * the same slots: an ICB's commands are read when its draws launch, so a
 * vertex barrier between one call's execution and the next call's encoding
 * is enough (render encoders are separated by full barriers already). One ICB
 * as big as the largest call, instead of room for every call. */
struct kk_icb_cmd {
   struct util_dynarray chunks; /* struct kk_icb_chunk, the last in use */
   bool executed;               /* an execution may still read the slots */
};

/* Parameters of kk_icb_encode, uploaded per call (matches the MSL struct) */
struct kk_icb_encode_params {
   uint64_t icb;          /* address of icb_id below, as an ICBContainer */
   uint64_t args;         /* the app's draw arguments */
   uint64_t count;        /* the app's count */
   uint64_t cond;         /* conditional rendering value, 0 for none */
   uint64_t range;        /* MTLIndirectCommandBufferExecutionRange below */
   uint64_t index_buffer; /* 0 for non-indexed draws */
   uint32_t stride;
   uint32_t max_draws;
   uint32_t location;
   uint32_t mode;         /* 0 draws, 1 16-bit indexed, 2 32-bit indexed */
   uint32_t prim;         /* MTLPrimitiveType */
   uint32_t cond_inverted;
   uint64_t icb_id;       /* the ICB's gpuResourceID */
   uint32_t range_location, range_length;
};

void kk_icb_device_init(struct kk_device *dev);
void kk_icb_device_finish(struct kk_device *dev);

/* An ICB of at least `count` slots for this command buffer's next call;
 * false when out of memory */
bool kk_icb_reserve(struct kk_cmd_buffer *cmd, uint32_t count,
                    struct kk_icb_chunk *chunk);

/* Back to the device's pool (the command buffer is no longer pending) */
void kk_icb_cmd_release(struct kk_device *dev, struct kk_icb_cmd *icb);

#endif
