#ifndef PAYLOAD_BROKER_HPP_
#define PAYLOAD_BROKER_HPP_

#pragma once

#include "BaseFilter.hpp"
#include <string>

/**
 * PayloadBroker is a base class for payload brokers.
 */
class PayloadBroker : public BaseFilter {
public:
  /**
   * This implementation extracts metadata of type NVDS_PAYLOAD_META
   * from the user metadata list on batch_meta and calls on_batch with each.
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);
  /**
   * Called by on_buffer when a distanceproto::Batch is found on the buffer.
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_payload(std::string* payload) = 0;
};

#endif  // PAYLOAD_BROKER_HPP_