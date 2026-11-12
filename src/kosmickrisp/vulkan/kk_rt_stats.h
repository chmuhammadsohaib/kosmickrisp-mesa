/*
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_RT_STATS_H
#define KK_RT_STATS_H 1

#include "kk_private.h"

#include "util/macros.h"

/* KK_RT_STATS=1: once per second, one stderr line with the ray tracing work
 * done in that second. Command counts are taken per command buffer while
 * recording and added when it is submitted (resubmissions re-record), so
 * they count executions. GPU times come from Metal 4 command buffer
 * timestamps around each vkCmdBuildAccelerationStructuresKHR (all its builds
 * and updates), copy and trace, taken once the work before them is done.
 * That needs the work in its own encoder, so with stats on these commands
 * end the current encoder and don't overlap with other work. Times are added
 * when the GPU has resolved them, so they can trail the counts by a frame.
 *
 * A second line, rt-mem, shows what could accumulate: live acceleration
 * structures and driver BOs, ray tracing BOs, residency set entries,
 * command buffers in flight and the Metal device's allocated size. */

struct kk_bo;
struct kk_cmd_buffer;
struct kk_device;

enum kk_rt_stat {
   KK_RT_STAT_FRAMES,
   KK_RT_STAT_BLAS_BUILDS,
   KK_RT_STAT_BLAS_BUILD_PRIMS,
   KK_RT_STAT_BLAS_UPDATES,
   KK_RT_STAT_BLAS_UPDATE_PRIMS,
   KK_RT_STAT_TLAS_BUILDS,
   KK_RT_STAT_TLAS_BUILD_INSTANCES,
   KK_RT_STAT_TLAS_UPDATES,
   KK_RT_STAT_TLAS_UPDATE_INSTANCES,
   KK_RT_STAT_AS_COPIES,
   KK_RT_STAT_AS_COMPACTIONS,
   KK_RT_STAT_TRACES,
   KK_RT_STAT_TRACES_INDIRECT,
   KK_RT_STAT_RAYS,
   KK_RT_STAT_PIPELINES,
   KK_RT_STAT_LIBRARIES,
   KK_RT_STAT_BUILD_CALLS,
   /* Resources, for the rt-mem line. Live counts are created - destroyed */
   KK_RT_STAT_RESIDENCY_ADDS,
   KK_RT_STAT_RESIDENCY_REMOVES,
   KK_RT_STAT_AS_PLACED_CREATED,
   KK_RT_STAT_AS_PLACED_DESTROYED,
   KK_RT_STAT_AS_POOLED_CREATED,
   KK_RT_STAT_AS_POOLED_DESTROYED,
   KK_RT_STAT_POOL_CHUNKS,
   KK_RT_STAT_POOL_BYTES,
   /* Updates done as builds (kk_as_rebuild_instead) */
   KK_RT_STAT_REBUILDS_SMALL,
   KK_RT_STAT_REBUILDS_REFIT_LIMIT,
   /* Operations not timed because every timestamp pair was in use */
   KK_RT_STAT_GPU_UNTIMED,
   KK_RT_STAT_BOS_CREATED,
   KK_RT_STAT_BOS_DESTROYED,
   KK_RT_STAT_BO_BYTES_CREATED,
   KK_RT_STAT_BO_BYTES_DESTROYED,
   KK_RT_STAT_SLOT_BOS,
   KK_RT_STAT_SLOT_BYTES,
   KK_RT_STAT_STACKS_CREATED,
   KK_RT_STAT_STACKS_DESTROYED,
   KK_RT_STAT_STACK_BYTES_CREATED,
   KK_RT_STAT_STACK_BYTES_DESTROYED,
   KK_RT_STAT_VFTS_CREATED,
   KK_RT_STAT_VFTS_DESTROYED,
   KK_RT_STAT_SUBMITS,
   KK_RT_STAT_VK_CMDS_SUBMITTED,
   KK_RT_STAT_CMDS_COMMITTED,
   KK_RT_STAT_CMDS_COMPLETED,
   KK_RT_STAT_COUNT,
};

enum kk_rt_stats_gpu {
   KK_RT_STATS_GPU_AS,
   KK_RT_STATS_GPU_TRACE,
   KK_RT_STATS_GPU_COUNT,
};

/* Counts of one command buffer recording */
struct kk_rt_stats_cmd {
   uint64_t counts[KK_RT_STAT_COUNT];
};

/* For counts kept outside command buffers. Use where struct kk_device is
 * complete. */
#define KK_RT_STATS_ADD(dev, stat, value)                                     \
   do {                                                                       \
      if (unlikely((dev)->rt_stats))                                          \
         kk_rt_stats_add((dev), (stat), (value));                             \
   } while (0)

#define KK_RT_STATS_ADD_BO(dev, bo, created)                                  \
   do {                                                                       \
      if (unlikely((dev)->rt_stats))                                          \
         kk_rt_stats_bo((dev), (bo), (created));                              \
   } while (0)

/* Sets dev->rt_stats, NULL unless KK_RT_STATS is set. Called before the
 * device creates any BO or residency entry, and finished after the last one
 * is gone, so the last line shows what the device leaked. */
VkResult kk_rt_stats_init(struct kk_device *dev);
void kk_rt_stats_finish(struct kk_device *dev);

/* Adds a submitted command buffer's counts */
void kk_rt_stats_add_cmd(struct kk_device *dev, struct kk_cmd_buffer *cmd);
void kk_rt_stats_add(struct kk_device *dev, enum kk_rt_stat stat,
                     uint64_t value);
/* Counts a BO created (or destroyed) by the device */
void kk_rt_stats_bo(struct kk_device *dev, const struct kk_bo *bo,
                    bool created);

/* Ends the current encoder and writes a timestamp; returns the pair to end
 * with kk_rt_stats_gpu_end, or -1 when no pair is free */
int32_t kk_rt_stats_gpu_begin(struct kk_cmd_buffer *cmd,
                              enum kk_rt_stats_gpu kind);
void kk_rt_stats_gpu_end(struct kk_cmd_buffer *cmd, int32_t pair);

#endif /* KK_RT_STATS_H */
