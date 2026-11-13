/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_sync.h"

#include <Metal/MTLEvent.h>

/* MTLFence */
mtl_fence *
mtl_new_fence(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_fence *)[dev newFence];
   }
}

/* MTLEvent */
mtl_event *
mtl_new_event(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newEvent];
   }
}

/* MTLSharedEvent */
mtl_shared_event *
mtl_new_shared_event(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newSharedEvent];
   }
}

int
mtl_shared_event_wait_until_signaled_value(mtl_shared_event *event_handle, uint64_t value, uint64_t timeout_ms)
{
   @autoreleasepool {
      id<MTLSharedEvent> event = (id<MTLSharedEvent>)event_handle;
      return (int)[event waitUntilSignaledValue:value timeoutMS:timeout_ms];
   }
}

/* Sleeps on a semaphore each event's listener signals at its value, instead of
 * polling every event in a loop (the common fallback spins a CPU core) */
int
mtl_shared_events_wait_any(mtl_shared_event **events, const uint64_t *values,
                           uint32_t count, uint64_t timeout_ms)
{
   static MTLSharedEventListener *listener;
   static dispatch_once_t once;
   dispatch_once(&once, ^{
      listener = [[MTLSharedEventListener alloc]
         initWithDispatchQueue:dispatch_queue_create("kk.sync.wait_any", NULL)];
   });

   @autoreleasepool {
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      for (uint32_t i = 0; i < count; i++) {
         id<MTLSharedEvent> event = (id<MTLSharedEvent>)events[i];
         if (event.signaledValue >= values[i])
            return (int)i;
         [event notifyListener:listener
                       atValue:values[i]
                         block:^(id<MTLSharedEvent> e, uint64_t v) {
                            dispatch_semaphore_signal(semaphore);
                         }];
      }
      const dispatch_time_t deadline =
         timeout_ms == UINT64_MAX
            ? DISPATCH_TIME_FOREVER
            : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ms * NSEC_PER_MSEC);
      for (;;) {
         for (uint32_t i = 0; i < count; i++) {
            if (((id<MTLSharedEvent>)events[i]).signaledValue >= values[i])
               return (int)i;
         }
         if (dispatch_semaphore_wait(semaphore, deadline))
            return -1;
      }
   }
}

void
mtl_shared_event_set_signaled_value(mtl_shared_event *event_handle, uint64_t value)
{
   @autoreleasepool {
      id<MTLSharedEvent> event = (id<MTLSharedEvent>)event_handle;
      event.signaledValue = value;
   }
}

uint64_t
mtl_shared_event_get_signaled_value(mtl_shared_event *event_handle)
{
   @autoreleasepool {
      id<MTLSharedEvent> event = (id<MTLSharedEvent>)event_handle;
      return event.signaledValue;
   }
}
