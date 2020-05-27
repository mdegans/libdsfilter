#ifndef TEST_CUDA_FILTER_HPP_
#define TEST_CUDA_FILTER_HPP_

#pragma once

#include "BaseCudaFilter.hpp"

/**
 * TestCudaFilter just logs info about the buffer.
 */
class TestCudaFilter : public BaseCudaFilter {
 public:
  /**
   * This class is for unit tests. It just passes the buffer through.
   *
   * Some info about the buffer is logged to the DEBUG level.
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);
};

#endif  // TEST_CUDA_FILTER_HPP_