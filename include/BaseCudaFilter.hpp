#ifndef BASE_CUDA_FILTER_HPP_
#define BASE_CUDA_FILTER_HPP_

#pragma once

// DeepStream includes:

#include <cuda_runtime.h>

#include "BaseFilter.hpp"

/**
 * Base class for filters needing a CUDA stream.
 */
class BaseCudaFilter: public BaseFilter {
 protected:
  // cuda stream to use for a filter
  cudaStream_t stream;

 public:
  BaseCudaFilter();
  virtual ~BaseCudaFilter();
};

#endif  // BASE_CUDA_FILTER_HPP_