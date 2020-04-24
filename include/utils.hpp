/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking
// this is derived from the CUDA samples but has been modified to use gstreamer
// error logging macros for consistency.

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cuda_runtime.h>
#include <gst/gst.h>

#pragma once

/**
 * Check the return type off a cuda function.
 * Return false on fail, true on success.
 */
#define checkCuda(val) __checkCuda((val), #val, __FILE__, __LINE__)
inline bool __checkCuda(cudaError_t err,
                        char const* const func,
                        const char* const file,
                        int const line) {
  if (err) {
    GST_ERROR("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
              static_cast<unsigned int>(err), cudaGetErrorName(err), func);
    return false;
  }
  return true;
}

/**
 * This will output the proper error string when calling cudaGetLastError
 * Return false on fail, true on success.
 */
#define checkLastCuda() __checkLastCuda(__FILE__, __LINE__)
inline bool __checkLastCuda(const char* file, const int line) {
  return __checkCuda(cudaGetLastError(), "(last)", file, line);
}

#endif  // UTILS_HPP_