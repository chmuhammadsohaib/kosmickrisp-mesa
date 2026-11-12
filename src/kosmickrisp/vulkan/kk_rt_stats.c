/*
 * SPDX-License-Identifier: MIT
 */

#include "kk_rt_stats.h"

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "c11/threads.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Timestamp pairs (begin, end) in flight. A pair whose timestamps never
 * resolve (its command buffer wasn't submitted) is reused after
 * KK_RT_STATS_EXPIRE_NS. */
#define KK_RT_STATS_PAIRS       2048u /* Metal allows 4096 entries */
#define KK_RT_STATS_UNRESOLVED  UINT64_MAX
#define KK_RT_STATS_EXPIRE_NS   (30ull * 1000000000ull)

/* kk_rt_stats::pair_kind values; kinds are enum kk_rt_stats_gpu + 1 */
#define KK_RT_STATS_PAIR_FREE 0u

struct kk_rt_stats {
   struct kk_device *dev;
   uint64_t totals[KK_RT_STAT_COUNT];

   thrd_t thread;
   mtx_t mutex;
   cnd_t cond;
   bool stop;

   /* GPU timing, NULL heap if Metal 4 timestamps aren't available */
   mtl_counter_heap *heap;
   struct kk_bo *bo; /* Resolved timestamps, two per pair */
   double ns_per_tick;
   uint32_t next_pair;
   uint32_t pair_kind[KK_RT_STATS_PAIRS];
   uint64_t pair_time_ns[KK_RT_STATS_PAIRS];
   /* Collected since the last line, by the reporting thread only */
   uint64_t ticks[KK_RT_STATS_GPU_COUNT];
};

static void
kk_rt_stats_collect_gpu(struct kk_rt_stats *stats, uint64_t now_ns,
                        uint64_t ticks[KK_RT_STATS_GPU_COUNT])
{
   const uint64_t *values = stats->bo->cpu;

   for (uint32_t p = 0; p < KK_RT_STATS_PAIRS; p++) {
      /* Free, or still being set up by kk_rt_stats_gpu_begin */
      const uint32_t kind = p_atomic_read(&stats->pair_kind[p]);
      if (kind == KK_RT_STATS_PAIR_FREE || kind > KK_RT_STATS_GPU_COUNT)
         continue;

      const uint64_t begin = p_atomic_read(&values[2 * p]);
      const uint64_t end = p_atomic_read(&values[2 * p + 1]);
      if (begin != KK_RT_STATS_UNRESOLVED && end != KK_RT_STATS_UNRESOLVED) {
         if (end > begin)
            ticks[kind - 1] += end - begin;
      } else if (now_ns - stats->pair_time_ns[p] < KK_RT_STATS_EXPIRE_NS) {
         continue;
      }

      p_atomic_set(&stats->pair_kind[p], KK_RT_STATS_PAIR_FREE);
   }
}

static void
kk_rt_stats_print(struct kk_rt_stats *stats, uint64_t printed[KK_RT_STAT_COUNT],
                  const char *label)
{
   uint64_t d[KK_RT_STAT_COUNT];
   for (unsigned i = 0; i < KK_RT_STAT_COUNT; i++) {
      const uint64_t total = p_atomic_read(&stats->totals[i]);
      d[i] = total - printed[i];
      printed[i] = total;
   }

   char gpu[128] = "GPU n/a";
   if (stats->heap) {
      snprintf(gpu, sizeof(gpu), "GPU AS %.3f ms trace %.3f ms (%" PRIu64
               " untimed)",
               stats->ticks[KK_RT_STATS_GPU_AS] * stats->ns_per_tick / 1e6,
               stats->ticks[KK_RT_STATS_GPU_TRACE] * stats->ns_per_tick / 1e6,
               d[KK_RT_STAT_GPU_UNTIMED]);
      memset(stats->ticks, 0, sizeof(stats->ticks));
   }

   fprintf(stderr,
           "[kk] %s: frames %" PRIu64
           " | BLAS build %" PRIu64 " (%" PRIu64 " prims)"
           " update %" PRIu64 " (%" PRIu64 " prims; %" PRIu64
           " small and %" PRIu64 " after the refit limit done as builds)"
           " | TLAS build %" PRIu64 " (%" PRIu64 " inst)"
           " update %" PRIu64 " (%" PRIu64 " inst)"
           " | %" PRIu64 " build calls"
           " | AS copy %" PRIu64 " compact %" PRIu64
           " | trace %" PRIu64 " (+%" PRIu64 " indirect) rays %" PRIu64
           " | pipelines %" PRIu64 " (+%" PRIu64 " libraries) | %s\n",
           label, d[KK_RT_STAT_FRAMES], d[KK_RT_STAT_BLAS_BUILDS],
           d[KK_RT_STAT_BLAS_BUILD_PRIMS], d[KK_RT_STAT_BLAS_UPDATES],
           d[KK_RT_STAT_BLAS_UPDATE_PRIMS], d[KK_RT_STAT_REBUILDS_SMALL],
           d[KK_RT_STAT_REBUILDS_REFIT_LIMIT], d[KK_RT_STAT_TLAS_BUILDS],
           d[KK_RT_STAT_TLAS_BUILD_INSTANCES], d[KK_RT_STAT_TLAS_UPDATES],
           d[KK_RT_STAT_TLAS_UPDATE_INSTANCES], d[KK_RT_STAT_BUILD_CALLS],
           d[KK_RT_STAT_AS_COPIES],
           d[KK_RT_STAT_AS_COMPACTIONS], d[KK_RT_STAT_TRACES],
           d[KK_RT_STAT_TRACES_INDIRECT], d[KK_RT_STAT_RAYS],
           d[KK_RT_STAT_PIPELINES], d[KK_RT_STAT_LIBRARIES], gpu);
}

#define MIB(bytes) ((double)(bytes) / (1024.0 * 1024.0))

static void
kk_rt_stats_print_mem(struct kk_rt_stats *stats,
                      uint64_t printed[KK_RT_STAT_COUNT], const char *label)
{
   struct kk_device *dev = stats->dev;
   uint64_t t[KK_RT_STAT_COUNT];
   for (unsigned i = 0; i < KK_RT_STAT_COUNT; i++)
      t[i] = p_atomic_read(&stats->totals[i]);

   const uint64_t residency_added =
      t[KK_RT_STAT_RESIDENCY_ADDS] - printed[KK_RT_STAT_RESIDENCY_ADDS];
   const uint64_t residency_removed =
      t[KK_RT_STAT_RESIDENCY_REMOVES] - printed[KK_RT_STAT_RESIDENCY_REMOVES];
   const uint64_t submits =
      t[KK_RT_STAT_SUBMITS] - printed[KK_RT_STAT_SUBMITS];
   const uint64_t vk_cmds =
      t[KK_RT_STAT_VK_CMDS_SUBMITTED] - printed[KK_RT_STAT_VK_CMDS_SUBMITTED];
   const uint64_t commits =
      t[KK_RT_STAT_CMDS_COMMITTED] - printed[KK_RT_STAT_CMDS_COMMITTED];
   const uint64_t placed_added =
      t[KK_RT_STAT_AS_PLACED_CREATED] - printed[KK_RT_STAT_AS_PLACED_CREATED];
   const uint64_t pooled_added =
      t[KK_RT_STAT_AS_POOLED_CREATED] - printed[KK_RT_STAT_AS_POOLED_CREATED];
   const uint64_t bos_added =
      t[KK_RT_STAT_BOS_CREATED] - printed[KK_RT_STAT_BOS_CREATED];
   const uint64_t bos_removed =
      t[KK_RT_STAT_BOS_DESTROYED] - printed[KK_RT_STAT_BOS_DESTROYED];
   memcpy(printed, t, sizeof(t));

   /* Re-recorded command buffers whose resources are freed when they
    * complete */
   const uint32_t pending =
      dev->has_queue ? p_atomic_read(&dev->queue.commits_in_flight) : 0;

   fprintf(stderr,
           "[kk] %s: AS live %" PRIu64 " (placed %" PRIu64 " +%" PRIu64
           ", pooled %" PRIu64 " +%" PRIu64 " in %" PRIu64 " heaps of %.1f MiB)"
           " | driver BOs live %" PRIu64 " (%.1f MiB) +%" PRIu64 " -%" PRIu64
           " | RT BOs: AS slots %" PRIu64 " (%.1f MiB), stacks %" PRIu64
           " (%.1f MiB), visible function tables %" PRIu64
           ", scratch/instance conversion 0 (app scratch)"
           " | residency %" PRIu64 " +%" PRIu64 " -%" PRIu64
           " | pending releases %u | cmd buffers in flight %" PRIu64
           " (this second: %" PRIu64 " submits of %" PRIu64
           " command buffers, %" PRIu64 " Metal command buffers)"
           " | MTLDevice allocated %.1f MiB\n",
           label,
           t[KK_RT_STAT_AS_PLACED_CREATED] - t[KK_RT_STAT_AS_PLACED_DESTROYED] +
              t[KK_RT_STAT_AS_POOLED_CREATED] -
              t[KK_RT_STAT_AS_POOLED_DESTROYED],
           t[KK_RT_STAT_AS_PLACED_CREATED] - t[KK_RT_STAT_AS_PLACED_DESTROYED],
           placed_added,
           t[KK_RT_STAT_AS_POOLED_CREATED] - t[KK_RT_STAT_AS_POOLED_DESTROYED],
           pooled_added, t[KK_RT_STAT_POOL_CHUNKS],
           MIB(t[KK_RT_STAT_POOL_BYTES]),
           t[KK_RT_STAT_BOS_CREATED] - t[KK_RT_STAT_BOS_DESTROYED],
           MIB(t[KK_RT_STAT_BO_BYTES_CREATED] -
               t[KK_RT_STAT_BO_BYTES_DESTROYED]),
           bos_added, bos_removed, t[KK_RT_STAT_SLOT_BOS],
           MIB(t[KK_RT_STAT_SLOT_BYTES]),
           t[KK_RT_STAT_STACKS_CREATED] - t[KK_RT_STAT_STACKS_DESTROYED],
           MIB(t[KK_RT_STAT_STACK_BYTES_CREATED] -
               t[KK_RT_STAT_STACK_BYTES_DESTROYED]),
           t[KK_RT_STAT_VFTS_CREATED] - t[KK_RT_STAT_VFTS_DESTROYED],
           t[KK_RT_STAT_RESIDENCY_ADDS] - t[KK_RT_STAT_RESIDENCY_REMOVES],
           residency_added, residency_removed, pending,
           t[KK_RT_STAT_CMDS_COMMITTED] - t[KK_RT_STAT_CMDS_COMPLETED],
           submits, vk_cmds, commits, MIB(mtl_device_current_allocated_size(dev->mtl_handle)));
}

static int
kk_rt_stats_thread(void *data)
{
   struct kk_rt_stats *stats = data;
   uint64_t printed[KK_RT_STAT_COUNT] = {0};
   uint64_t printed_mem[KK_RT_STAT_COUNT] = {0};

   /* Collects timestamps every 100 ms so pairs are free again quickly, and
    * prints every second */
   uint64_t next_print_ns = os_time_get_nano() + 1000000000ull;
   mtx_lock(&stats->mutex);
   while (!stats->stop) {
      struct timespec deadline;
      timespec_get(&deadline, TIME_UTC);
      deadline.tv_nsec += 100000000;
      if (deadline.tv_nsec >= 1000000000) {
         deadline.tv_sec++;
         deadline.tv_nsec -= 1000000000;
      }
      while (!stats->stop &&
             cnd_timedwait(&stats->cond, &stats->mutex, &deadline) ==
                thrd_success)
         ;
      if (stats->stop)
         break;

      const uint64_t now_ns = os_time_get_nano();
      if (stats->heap)
         kk_rt_stats_collect_gpu(stats, now_ns, stats->ticks);
      if (now_ns < next_print_ns)
         continue;
      next_print_ns += 1000000000ull;

      kk_rt_stats_print(stats, printed, "rt-stats");
      kk_rt_stats_print_mem(stats, printed_mem, "rt-mem");
   }
   mtx_unlock(&stats->mutex);

   /* What happened since the last line, so the lines add up to the totals */
   if (stats->heap)
      kk_rt_stats_collect_gpu(stats, os_time_get_nano(), stats->ticks);
   kk_rt_stats_print(stats, printed, "rt-stats (until device destroyed)");
   kk_rt_stats_print_mem(stats, printed_mem, "rt-mem (device destroyed)");

   return 0;
}

static void
kk_rt_stats_init_gpu(struct kk_device *dev, struct kk_rt_stats *stats)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   stats->heap =
      mtl_new_timestamp_counter_heap(dev->mtl_handle, 2u * KK_RT_STATS_PAIRS);
   if (!stats->heap)
      return;

   if (kk_alloc_bo(dev, &dev->vk.base,
                   2u * KK_RT_STATS_PAIRS * sizeof(uint64_t), 8u,
                   &stats->bo) != VK_SUCCESS) {
      mtl_release(stats->heap);
      stats->heap = NULL;
      return;
   }

   stats->ns_per_tick = pdev->vk.properties.timestampPeriod;
}

VkResult
kk_rt_stats_init(struct kk_device *dev)
{
   dev->rt_stats = NULL;
   if (!debug_get_bool_option("KK_RT_STATS", false))
      return VK_SUCCESS;

   struct kk_rt_stats *stats =
      vk_zalloc(&dev->vk.alloc, sizeof(*stats), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!stats)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   stats->dev = dev;

   kk_rt_stats_init_gpu(dev, stats);

   mtx_init(&stats->mutex, mtx_plain);
   cnd_init(&stats->cond);
   if (thrd_create(&stats->thread, kk_rt_stats_thread, stats) !=
       thrd_success) {
      fprintf(stderr, "[kk] KK_RT_STATS: can't start the reporting thread\n");
      cnd_destroy(&stats->cond);
      mtx_destroy(&stats->mutex);
      if (stats->bo)
         kk_destroy_bo(dev, stats->bo);
      if (stats->heap)
         mtl_release(stats->heap);
      vk_free(&dev->vk.alloc, stats);
      return VK_SUCCESS;
   }

   fprintf(stderr, "[kk] KK_RT_STATS: on, GPU times %s\n",
           stats->heap ? "from Metal 4 timestamps" : "not available");
   dev->rt_stats = stats;
   return VK_SUCCESS;
}

void
kk_rt_stats_finish(struct kk_device *dev)
{
   struct kk_rt_stats *stats = dev->rt_stats;
   if (!stats)
      return;

   mtx_lock(&stats->mutex);
   stats->stop = true;
   cnd_signal(&stats->cond);
   mtx_unlock(&stats->mutex);
   thrd_join(stats->thread, NULL);

   /* Its own BO isn't counted */
   dev->rt_stats = NULL;
   cnd_destroy(&stats->cond);
   mtx_destroy(&stats->mutex);
   if (stats->bo)
      kk_destroy_bo(dev, stats->bo);
   if (stats->heap)
      mtl_release(stats->heap);
   vk_free(&dev->vk.alloc, stats);
}

void
kk_rt_stats_add(struct kk_device *dev, enum kk_rt_stat stat, uint64_t value)
{
   p_atomic_add(&dev->rt_stats->totals[stat], value);
}

void
kk_rt_stats_bo(struct kk_device *dev, const struct kk_bo *bo, bool created)
{
   uint64_t *totals = dev->rt_stats->totals;
   p_atomic_inc(&totals[created ? KK_RT_STAT_BOS_CREATED
                                : KK_RT_STAT_BOS_DESTROYED]);
   p_atomic_add(&totals[created ? KK_RT_STAT_BO_BYTES_CREATED
                                : KK_RT_STAT_BO_BYTES_DESTROYED],
                bo->size_B);
}

void
kk_rt_stats_add_cmd(struct kk_device *dev, struct kk_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < KK_RT_STAT_COUNT; i++) {
      if (cmd->rt_stats.counts[i])
         p_atomic_add(&dev->rt_stats->totals[i], cmd->rt_stats.counts[i]);
   }
}

int32_t
kk_rt_stats_gpu_begin(struct kk_cmd_buffer *cmd, enum kk_rt_stats_gpu kind)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_rt_stats *stats = dev->rt_stats;
   if (!stats->heap)
      return -1;

   const uint32_t pair =
      p_atomic_inc_return(&stats->next_pair) % KK_RT_STATS_PAIRS;
   if (p_atomic_cmpxchg(&stats->pair_kind[pair], KK_RT_STATS_PAIR_FREE,
                        UINT32_MAX) != KK_RT_STATS_PAIR_FREE) {
      kk_rt_stats_add(dev, KK_RT_STAT_GPU_UNTIMED, 1);
      return -1;
   }

   uint64_t *values = stats->bo->cpu;
   values[2 * pair] = KK_RT_STATS_UNRESOLVED;
   values[2 * pair + 1] = KK_RT_STATS_UNRESOLVED;
   stats->pair_time_ns[pair] = os_time_get_nano();
   /* The reporter skips the pair until this, after the sentinels */
   p_atomic_set(&stats->pair_kind[pair], kind + 1);

   /* Timestamps between encoders come after all the work before them.
    * Timestamps in a compute encoder don't wait for acceleration structure
    * work at all (a 35 ms build measured 0). cs_get_compute deals with an
    * open render encoder. */
   if (cmd->metal.render)
      cs_get_compute(cmd);
   cs_end(cmd);
   mtl_command_buffer_write_timestamp(cmd->metal.cmd_buf, stats->heap,
                                      2 * pair);
   return pair;
}

void
kk_rt_stats_gpu_end(struct kk_cmd_buffer *cmd, int32_t pair)
{
   if (pair < 0)
      return;

   struct kk_rt_stats *stats = kk_cmd_buffer_device(cmd)->rt_stats;
   cs_end(cmd);
   mtl_command_buffer_write_timestamp(cmd->metal.cmd_buf, stats->heap,
                                      2 * pair + 1);
   mtl_command_resolve_counter_heap(cmd->metal.cmd_buf, stats->heap, 2 * pair,
                                    2, stats->bo->gpu + 2 * pair *
                                                           sizeof(uint64_t));
}
