#ifndef HASH_CUDA_FILTER_HPP_
#define HASH_CUDA_FILTER_HPP_

#pragma once

#include "BaseCudaFilter.hpp"
#include "nvbufsurftransform.h"

/**
 *  frame level hash user meta type
 */
#define HASH_USER_FRAME_META (nvds_get_user_meta_type((gchar*)"NVIDIA.NVINFER.USER_META"))

/**
 * HashCudaFilter hashes.
 */
class HashCudaFilter : public BaseCudaFilter {
protected:
  /** The per-session transform config */
  NvBufSurfTransformConfigParams* config;
  /** The per-frame transform config */
  NvBufSurfTransformParams* params;

  /** A temporary surface */
  NvBufSurface* tmp_surface;
  /** creation parameters for the temp surface **/
  NvBufSurfaceCreateParams* create_params;
  /**
   * Resets the temporary surface
   */
  virtual bool create_tmp_surface(const NvBufSurface* surf);
  /**
   * Sets the NvBufSurfTransformRect lists appropriately.
   */
  virtual bool set_tx_rects(NvBufSurface* surf);
public:
  HashCudaFilter();
  virtual ~HashCudaFilter();
  /**
   * Called on every NVMM batched buffer.
   *
   * calculates a hash for each frame in the the batch and attaches it
   *  
   *
   * return a GstFlowReturn (success, failure, etc.)
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);
  /**
   * Hash a batch of frames and attach the hash as HASH_USER_BATCH_META.
   * 
   * return false on failure.
   */
  virtual bool on_batch(NvBufSurface* surf, NvDsBatchMeta* batch_meta);
  /**
   * Hash height (in pixels).
   */
  static const int HEIGHT=8;
  /**
   * Hash width (in pixels).
   */
  static const int WIDTH=8;
  /**
   * Size of a hash in bytes.
   */
  static const int SIZE=WIDTH*HEIGHT;
};

#endif  // TEST_CUDA_FILTER_HPP_