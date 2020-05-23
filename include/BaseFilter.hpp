#ifndef BASE_FILTER_HPP_
#define BASE_FILTER_HPP_

#pragma once

// DeepStream includes:

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

/**
 * Base class for all filters
 */
class BaseFilter {
 public:
  virtual ~BaseFilter() = default;
  /**
   * Called on every NVMM batched buffer.
   *
   * Should be connected to a buffer callback or used in a filter plugin.
   *
   * return a GstFlowReturn (success, failure, etc.)
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);

  /**
   * Called on every frame by on_buffer
   *
   * return true on success, false on failure
   */
  virtual bool on_frame(NvBufSurface* surf, NvDsFrameMeta* frame_meta);

  /**
   * Called on every object meta by on_frame
   *
   * return true on success, false on failure
   */
  virtual bool on_object(NvDsFrameMeta* f_meta,
                         NvDsObjectMeta* o_meta,
                         NvBufSurfaceParams* frame);
};

#endif  // BASE_FILTER_HPP_