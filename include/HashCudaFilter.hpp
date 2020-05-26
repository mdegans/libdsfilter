#ifndef HASH_CUDA_FILTER_HPP_
#define HASH_CUDA_FILTER_HPP_

#pragma once

#include "BaseCudaFilter.hpp"

typedef char Hash[64];
typedef char SmHash[16];

/**
 * HashCudaFilter hashes.
 */
class HashCudaFilter : public BaseCudaFilter {
private:
  Hash& hash;
  SmHash& sm_hash;
public:
  HashCudaFilter();
  virtual ~HashCudaFilter() = default;
  /**
   * on_frame calculates a simple image hash for each frame
   */
  bool on_frame(NvBufSurface* surf, NvDsFrameMeta* frame_meta);

  const Hash& getHash() const { return hash; };
  const SmHash& getSmHash() const { return sm_hash; }
};

#endif  // TEST_CUDA_FILTER_HPP_