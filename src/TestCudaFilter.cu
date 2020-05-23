#include "TestCudaFilter.hpp"

GstFlowReturn TestCudaFilter::on_buffer(GstBuffer* buf) {
  GST_DEBUG_OBJECT(buf, "Got buffer.");

  return GST_FLOW_OK;
}
