#ifndef TEST_CUDA_FILTER_HPP_
#define TEST_CUDA_FILTER_HPP_

#pragma once

#include "BaseCudaFilter.hpp"

/**
 * TestCudaFilter just prints some information about the objects it receives
 */
class TestCudaFilter : public BaseCudaFilter {
public:
	/**
	 * Implementation of on_object that just prints metadata information
	 */
	bool on_object(NvDsFrameMeta * f_meta, NvDsObjectMeta * o_meta, NvBufSurfaceParams * frame);
};

#endif // TEST_CUDA_FILTER_HPP_