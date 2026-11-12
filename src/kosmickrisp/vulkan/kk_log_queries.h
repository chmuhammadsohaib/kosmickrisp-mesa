/*
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_LOG_QUERIES_H
#define KK_LOG_QUERIES_H 1

#include "vulkan/vulkan_core.h"

#include "util/u_debug.h"
#include "vk_enum_to_str.h"

#include <stdbool.h>
#include <stdio.h>

/* KK_LOG_QUERIES=1 prints to stderr what an application queries about the
 * device and what it enables at vkCreateDevice. The struct printers are
 * generated from vk.xml by kk_log_queries_gen.py. */

/* Prints every field of a structure, picked by its sType */
void kk_log_struct(const char *prefix, const void *data);
void kk_log_features(const char *prefix,
                     const VkPhysicalDeviceFeatures *features);
void kk_log_properties(const char *prefix,
                       const VkPhysicalDeviceProperties *properties);

static inline bool
kk_log_queries_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("KK_LOG_QUERIES", false);
   return enabled;
}

/* Prints the sType of every structure in a pNext chain */
static inline void
kk_log_chain_types(const char *prefix, const void *chain)
{
   unsigned i = 0;
   for (const VkBaseInStructure *s = chain; s; s = s->pNext, i++)
      fprintf(stderr, "[kk] %s pNext[%u] = %s\n", prefix, i,
              vk_StructureType_to_str(s->sType));
}

/* Prints every field of every structure in a pNext chain */
static inline void
kk_log_chain(const char *prefix, const void *chain)
{
   for (const VkBaseInStructure *s = chain; s; s = s->pNext)
      kk_log_struct(prefix, s);
}

#endif /* KK_LOG_QUERIES_H */
