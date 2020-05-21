#include "BaseCudaFilter.hpp"

#include "utils.hpp"

// constructor and destructor:

BaseCudaFilter::BaseCudaFilter() {
  GST_DEBUG("BaseCudaFilter constructor. Creating CUDA stream.");
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}
BaseCudaFilter::~BaseCudaFilter() {
  GST_DEBUG("BaseCudaFilter destroy. Destroying CUDA stream.");
  checkCuda(cudaStreamDestroy(stream));
}

