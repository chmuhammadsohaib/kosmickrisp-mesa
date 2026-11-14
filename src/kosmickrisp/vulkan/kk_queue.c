/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_queue.h"
#include "kk_rt_stats.h"

#include "kk_sparse.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

struct kk_commit_data {
   struct kk_queue *queue;
   struct kk_cmd_buffer *cmd;
};

/* KK_GPU_TIMELINE=1: once a second, how busy the GPU was across all queues,
 * how busy each queue was, and how long committed work waited before the GPU
 * started it (event waits, other queues, scheduling), from Metal's commit
 * feedback. For finding frames that wait instead of render. */
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

#define KK_TL_MAX_QUEUES 8
#define KK_TL_MAX_SPANS 8192

struct kk_tl_commit {
   mtl_feedback_handler_callback callback;
   void *user_data;
   uint32_t queue;
   uint32_t waits;
   double committed;
};

static struct {
   int enabled; /* -1: not read yet */
   mtx_t lock;
   const void *queues[KK_TL_MAX_QUEUES];
   double window;
   uint32_t spans;
   double start[KK_TL_MAX_SPANS], end[KK_TL_MAX_SPANS];
   double queue_busy[KK_TL_MAX_QUEUES];
   uint32_t queue_commits[KK_TL_MAX_QUEUES], queue_waiting[KK_TL_MAX_QUEUES];
   double latency_sum, latency_max;
   uint32_t commits;
} kk_tl = {.enabled = -1};

static uint32_t kk_tl_draws[4];
uint32_t *kk_tl_draw_counters;
static bool kk_tl_enabled(void);

/* libkk_count_draws' totals, in a buffer the GPU adds to */
uint64_t kk_tl_draw_stats_gpu;
static uint32_t *kk_tl_draw_stats;

void
kk_tl_init_device(struct kk_device *dev)
{
   static struct kk_bo *bo;
   if (!kk_tl_enabled() || bo)
      return;
   if (kk_alloc_bo(dev, &dev->vk.base, 64u, 0, &bo) == VK_SUCCESS) {
      memset(bo->cpu, 0, 64u);
      kk_tl_draw_stats = bo->cpu;
      kk_tl_draw_stats_gpu = bo->gpu;
   }
}

static double
kk_tl_now(void)
{
   static mach_timebase_info_data_t tb;
   if (!tb.denom)
      mach_timebase_info(&tb);
   return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static bool
kk_tl_enabled(void)
{
   if (unlikely(kk_tl.enabled < 0)) {
      const char *v = getenv("KK_GPU_TIMELINE");
      mtx_init(&kk_tl.lock, mtx_plain);
      kk_tl.enabled = v && v[0] == '1';
      if (kk_tl.enabled)
         kk_tl_draw_counters = kk_tl_draws;
   }
   return kk_tl.enabled;
}

static int
kk_tl_cmp(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

/* Union of the window's GPU spans; sorts by start, carrying ends along */
static double
kk_tl_union(void)
{
   double pairs[KK_TL_MAX_SPANS][2];
   for (uint32_t i = 0; i < kk_tl.spans; i++) {
      pairs[i][0] = kk_tl.start[i];
      pairs[i][1] = kk_tl.end[i];
   }
   qsort(pairs, kk_tl.spans, sizeof(pairs[0]), kk_tl_cmp);
   double busy = 0, s = 0, e = -1;
   for (uint32_t i = 0; i < kk_tl.spans; i++) {
      if (pairs[i][0] > e) {
         if (e > s)
            busy += e - s;
         s = pairs[i][0];
         e = pairs[i][1];
      } else if (pairs[i][1] > e) {
         e = pairs[i][1];
      }
   }
   if (e > s)
      busy += e - s;
   return busy;
}

static void
kk_tl_report_locked(void)
{
   if (!kk_tl.commits)
      return;
   fprintf(stderr, "[kk] gpu-timeline: busy %.0f%% | %u commits, waited %.2f ms avg %.2f ms max before starting |",
           100.0 * kk_tl_union(), kk_tl.commits, 1e3 * kk_tl.latency_sum / kk_tl.commits,
           1e3 * kk_tl.latency_max);
   for (uint32_t q = 0; q < KK_TL_MAX_QUEUES && kk_tl.queues[q]; q++)
      fprintf(stderr, " q%u %.0f%% (%u, %u with waits)", q, 100.0 * kk_tl.queue_busy[q],
              kk_tl.queue_commits[q], kk_tl.queue_waiting[q]);
   fprintf(stderr, " | Metal draws %u, %u indirect-count calls expanded to %u, %u GPU-built",
           p_atomic_xchg(&kk_tl_draws[0], 0), p_atomic_xchg(&kk_tl_draws[1], 0),
           p_atomic_xchg(&kk_tl_draws[2], 0), p_atomic_xchg(&kk_tl_draws[3], 0));
   if (kk_tl_draw_stats)
      fprintf(stderr, ", real draws %u (largest call %u, %u calls)",
              p_atomic_xchg(&kk_tl_draw_stats[0], 0), p_atomic_xchg(&kk_tl_draw_stats[1], 0),
              p_atomic_xchg(&kk_tl_draw_stats[2], 0));
   fprintf(stderr, "\n");
   memset(kk_tl.queue_busy, 0, sizeof(kk_tl.queue_busy));
   memset(kk_tl.queue_commits, 0, sizeof(kk_tl.queue_commits));
   memset(kk_tl.queue_waiting, 0, sizeof(kk_tl.queue_waiting));
   kk_tl.spans = kk_tl.commits = 0;
   kk_tl.latency_sum = kk_tl.latency_max = 0;
}

static void
kk_tl_feedback(struct mtl_feedback_data *data)
{
   struct kk_tl_commit *c = data->user_data;
   mtx_lock(&kk_tl.lock);
   if (data->gpu_end > kk_tl.window + 1.0) {
      kk_tl_report_locked();
      kk_tl.window = data->gpu_end;
   }
   if (data->gpu_end > data->gpu_start && data->gpu_start > 0) {
      double start = MAX2(data->gpu_start, kk_tl.window);
      if (kk_tl.spans < KK_TL_MAX_SPANS && data->gpu_end > start) {
         kk_tl.start[kk_tl.spans] = start;
         kk_tl.end[kk_tl.spans++] = data->gpu_end;
         kk_tl.queue_busy[c->queue] += data->gpu_end - start;
      }
      double latency = data->gpu_start - c->committed;
      if (latency > 0) {
         kk_tl.latency_sum += latency;
         kk_tl.latency_max = MAX2(kk_tl.latency_max, latency);
      }
   }
   kk_tl.commits++;
   kk_tl.queue_commits[c->queue]++;
   kk_tl.queue_waiting[c->queue] += c->waits > 0;
   mtx_unlock(&kk_tl.lock);

   data->user_data = c->user_data;
   c->callback(data);
   free(c);
}

static uint32_t
kk_tl_queue_index(const struct kk_queue *queue)
{
   for (uint32_t q = 0; q < KK_TL_MAX_QUEUES; q++) {
      if (kk_tl.queues[q] == queue)
         return q;
      if (!kk_tl.queues[q]) {
         kk_tl.queues[q] = queue;
         return q;
      }
   }
   return KK_TL_MAX_QUEUES - 1;
}

static void
kk_queue_wait_for_commits(struct kk_queue *queue)
{
   mtx_lock(&queue->mutex);
   while (queue->commits_in_flight > 0u)
      cnd_wait(&queue->cond, &queue->mutex);
   mtx_unlock(&queue->mutex);
}

static void
kk_queue_release_cmd_buffer_locked(struct kk_queue *queue,
                                   struct kk_cmd_buffer *cmd)
{
   kk_cmd_buffer_ops.reset(&cmd->vk, 0u);
   util_dynarray_append(&queue->free_cmd_buffers, &cmd->vk);
}

/* KK_SYNC_DEBUG: the last semaphore waits and signals the queues submitted,
 * so a GPU timeout can say which wait was never satisfied (a queue waiting on
 * an event nothing signals times out like a hung shader). Events are retained
 * while they're in the ring. */
#define KK_SYNC_RING 256
static struct {
   simple_mtx_t lock;
   bool init;
   uint32_t next;
   struct {
      mtl_shared_event *event;
      uint64_t value;
      const void *queue;
      bool signal;
   } rec[KK_SYNC_RING];
} kk_sync_ring;

static void
kk_sync_ring_add(const void *queue, mtl_shared_event *event, uint64_t value,
                 bool signal)
{
   simple_mtx_lock(&kk_sync_ring.lock);
   uint32_t i = kk_sync_ring.next++ % KK_SYNC_RING;
   if (kk_sync_ring.rec[i].event)
      mtl_release(kk_sync_ring.rec[i].event);
   kk_sync_ring.rec[i].event = mtl_retain(event);
   kk_sync_ring.rec[i].value = value;
   kk_sync_ring.rec[i].queue = queue;
   kk_sync_ring.rec[i].signal = signal;
   simple_mtx_unlock(&kk_sync_ring.lock);
}

static void
kk_sync_ring_report(void)
{
   simple_mtx_lock(&kk_sync_ring.lock);
   unsigned unsatisfied = 0;
   for (uint32_t i = 0; i < KK_SYNC_RING; i++) {
      if (!kk_sync_ring.rec[i].event || kk_sync_ring.rec[i].signal)
         continue;
      uint64_t now = mtl_shared_event_get_signaled_value(kk_sync_ring.rec[i].event);
      if (now >= kk_sync_ring.rec[i].value)
         continue;
      uint64_t submitted = 0;
      const void *signaller = NULL;
      for (uint32_t j = 0; j < KK_SYNC_RING; j++) {
         if (kk_sync_ring.rec[j].signal &&
             kk_sync_ring.rec[j].event == kk_sync_ring.rec[i].event &&
             kk_sync_ring.rec[j].value > submitted) {
            submitted = kk_sync_ring.rec[j].value;
            signaller = kk_sync_ring.rec[j].queue;
         }
      }
      if (unsatisfied++ < 16)
         fprintf(stderr,
                 "[kk] sync-debug: queue %p waits on event %p for %llu, which is at %llu; "
                 "highest signal submitted %llu (queue %p)\n",
                 kk_sync_ring.rec[i].queue, kk_sync_ring.rec[i].event,
                 (unsigned long long)kk_sync_ring.rec[i].value, (unsigned long long)now,
                 (unsigned long long)submitted, signaller);
   }
   fprintf(stderr, "[kk] sync-debug: %u unsatisfied waits among the last %u\n", unsatisfied,
           MIN2(kk_sync_ring.next, KK_SYNC_RING));
   /* The last submissions in order, to see whether the app's own order is a cycle */
   uint32_t n = MIN2(kk_sync_ring.next, 48u);
   for (uint32_t k = kk_sync_ring.next - n; k < kk_sync_ring.next; k++) {
      uint32_t i = k % KK_SYNC_RING;
      fprintf(stderr, "[kk] sync-order %u: queue %p %s event %p %llu (now %llu)\n", k,
              kk_sync_ring.rec[i].queue, kk_sync_ring.rec[i].signal ? "signal" : "wait  ",
              kk_sync_ring.rec[i].event, (unsigned long long)kk_sync_ring.rec[i].value,
              (unsigned long long)mtl_shared_event_get_signaled_value(kk_sync_ring.rec[i].event));
   }
   simple_mtx_unlock(&kk_sync_ring.lock);
}

/* KK_SYNC_DEBUG: what each committed command buffer recorded, reported when
 * the GPU fails one (a timeout names the work that ran too long) */
struct kk_sd_commit {
   mtl_feedback_handler_callback callback;
   void *user_data;
   const struct kk_queue *queue;
   struct kk_sd_counts {
      uint32_t draws, icb_calls, dispatches, traces, as_builds, mfx_scales,
         mfx_interpolations;
   } counts;
};

static void
kk_sd_feedback(struct mtl_feedback_data *data)
{
   struct kk_sd_commit *c = data->user_data;
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      static uint32_t reports;
      if (p_atomic_inc_return(&reports) <= 4)
         fprintf(stderr,
                 "[kk] sync-debug: failed command buffer on queue %p: %u draws, %u GPU-built draw calls, "
                 "%u dispatches, %u ray dispatches, %u acceleration structure builds, %u MetalFX upscales, "
                 "%u MetalFX interpolations\n",
                 c->queue, c->counts.draws, c->counts.icb_calls, c->counts.dispatches, c->counts.traces,
                 c->counts.as_builds, c->counts.mfx_scales, c->counts.mfx_interpolations);
   }
   data->user_data = c->user_data;
   c->callback(data);
   free(c);
}

static void
check_device_lost(struct kk_device *dev, struct mtl_feedback_data *data)
{
   if (unlikely(dev->sync_debug.enabled)) {
      p_atomic_inc(&dev->sync_debug.completed);
      if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE)
         dev->sync_debug.last_error = data->error;
   }
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      /* Games often just wait forever once the device is lost; say so once */
      static bool reported;
      if (!p_atomic_xchg(&reported, true)) {
         fprintf(stderr, "[kk] GPU error: %s (%s); the device is lost\n",
                 mtl_command_queue_error_to_string(data->error),
                 data->error_message ? data->error_message : "no message");
         kk_icb_debug_report(dev);
         if (dev->atomic64_locks)
            fprintf(stderr, "[kk] 64-bit atomics: %u lanes had given up on a lock\n",
                    p_atomic_read(&((uint32_t *)dev->atomic64_locks->cpu)[KK_ATOMIC64_GIVE_UPS]));
         if (dev->sync_debug.enabled && kk_sync_ring.init)
            kk_sync_ring_report();
      }
      vk_device_set_lost(
         &dev->vk, "Command queue error: %s, with message \"%s\"",
         mtl_command_queue_error_to_string(data->error), data->error_message);
   }
}

static void
commit_callback(struct mtl_feedback_data *data)
{
   struct kk_device *dev = (struct kk_device *)data->user_data;
   check_device_lost(dev, data);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_CMDS_COMPLETED, 1);
}

static void
rerecord_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_commit_data *commit = (struct kk_commit_data *)data->user_data;
   struct kk_queue *queue = commit->queue;
   struct kk_device *dev = kk_queue_device(queue);

   check_device_lost(dev, data);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_CMDS_COMPLETED, 1);

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   kk_queue_release_cmd_buffer_locked(queue, commit->cmd);

   assert(queue->commits_in_flight > 0u);
   if (--queue->commits_in_flight == 0u) {
      kk_ResetCommandPool(kk_device_to_handle(dev),
                          kk_cmd_pool_to_handle(queue->cmd_pool), 0u);
      cnd_broadcast(&queue->cond);
   }

   mtx_unlock(&queue->mutex);

   vk_free(&dev->vk.alloc, commit);
}

static bool
kk_cmd_buffer_has_work(struct kk_cmd_buffer *cmd)
{
   return cmd->metal.cmd_buf;
}

static void
kk_queue_commit(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                mtl_feedback_handler_callback callback, void *user_data)
{
   assert(kk_cmd_buffer_has_work(cmd));

   KK_RT_STATS_ADD(kk_queue_device(queue), KK_RT_STAT_CMDS_COMMITTED, 1);
   if (unlikely(kk_queue_device(queue)->sync_debug.enabled))
      p_atomic_inc(&kk_queue_device(queue)->sync_debug.committed);

   if (unlikely(kk_queue_device(queue)->sync_debug.enabled)) {
      struct kk_sd_commit *c = malloc(sizeof(*c));
      if (c) {
         *c = (struct kk_sd_commit){callback, user_data, queue};
         STATIC_ASSERT(sizeof(c->counts) == sizeof(cmd->counts));
         memcpy(&c->counts, &cmd->counts, sizeof(c->counts));
         callback = kk_sd_feedback;
         user_data = c;
      }
   }

   if (unlikely(kk_tl_enabled())) {
      struct kk_tl_commit *c = malloc(sizeof(*c));
      if (c) {
         mtx_lock(&kk_tl.lock);
         *c = (struct kk_tl_commit){callback, user_data, kk_tl_queue_index(queue),
                                    queue->timeline_waits, kk_tl_now()};
         mtx_unlock(&kk_tl.lock);
         queue->timeline_waits = 0;
         callback = kk_tl_feedback;
         user_data = c;
      }
   }

   mtl_commit_options *options = mtl_new_commit_options();
   mtl_commit_options_add_feedback_handler(options, callback, user_data);

   mtl_command_queue_commit(queue->mtl_handle, &cmd->metal.cmd_buf, 1u,
                            options);
   mtl_release(options);
}

static VkResult
rerecord_and_commit_cmd_buffer(struct kk_queue *queue,
                               struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);

   if (unlikely(dev->rt_stats))
      kk_rt_stats_add_cmd(dev, rerecord);

   queue->commits_in_flight++;
   kk_queue_commit(queue, rerecord, rerecord_commit_callback, commit);

   mtx_unlock(&queue->mutex);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   KK_RT_STATS_ADD(dev, KK_RT_STAT_SUBMITS, 1);
   KK_RT_STATS_ADD(dev, KK_RT_STAT_VK_CMDS_SUBMITTED,
                   submit->command_buffer_count);

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
      if (unlikely(dev->sync_debug.enabled))
         kk_sync_ring_add(queue, sync->mtl_handle, wait->wait_value, false);
   }
   /* The first commit of this submit is the one that waits */
   queue->timeline_waits = submit->wait_count;

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   /* Sparse binds come after the waits and before the work, in queue order;
    * Metal applies the mapping updates in its queue's order too. */
   if (vk_queue_submit_has_bind(submit))
      kk_queue_bind_sparse(queue, submit);

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal's command buffers are one time use, re-record multiple
       * submissions. */
      if (cmd_buffer->submitted) {
         VkResult result = rerecord_and_commit_cmd_buffer(queue, cmd_buffer);
         if (result != VK_SUCCESS)
            return result;
      } else if (kk_cmd_buffer_has_work(cmd_buffer)) {
         if (unlikely(dev->rt_stats))
            kk_rt_stats_add_cmd(dev, cmd_buffer);
         kk_queue_commit(queue, cmd_buffer, commit_callback, dev);
      }

      cmd_buffer->submitted = true;

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }

   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
      if (unlikely(dev->sync_debug.enabled))
         kk_sync_ring_add(queue, sync->mtl_handle, signal->signal_value, true);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   if (!kk_sync_ring.init) {
      simple_mtx_init(&kk_sync_ring.lock, mtx_plain);
      kk_sync_ring.init = true;
   }

   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pCreateInfo->queueFamilyIndex,
   };
   VkCommandPool pool_handle;
   result = kk_CreateCommandPool(kk_device_to_handle(dev), &pool_info, NULL,
                                 &pool_handle);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->cmd_pool = kk_cmd_pool_from_handle(pool_handle);

   queue->free_cmd_buffers = UTIL_DYNARRAY_INIT;
   queue->commits_in_flight = 0u;
   mtx_init(&queue->mutex, mtx_plain);
   cnd_init(&queue->cond);

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);

   queue->vk.driver_submit = kk_queue_submit;

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   /* There may be in-flight command buffers, wait for them to finish. */
   kk_queue_wait_for_commits(queue);

   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->mtl_handle);

   kk_DestroyCommandPool(kk_device_to_handle(dev),
                         kk_cmd_pool_to_handle(queue->cmd_pool), NULL);
   util_dynarray_fini(&queue->free_cmd_buffers);
   cnd_destroy(&queue->cond);
   mtx_destroy(&queue->mutex);

   vk_queue_finish(&queue->vk);
}
