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
  virtual ~PayloadBroker() = default;
  /**
   * This implementation extracts metadata of type NVDS_PAYLOAD_META
   * from the user metadata list on batch_meta and calls on_batch with each.
   * 
   * note: the default implementation locks and unlocks the metadata, so care
   *  should be taken to not acquire the lock again (deadloack) and not to do
   *  anything blocking in the methods it calls below.
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);
  /**
   * Called by on_buffer when a serialized payload is found on the buffer.
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_payload(std::string* payload) = 0;
};

#endif  // PAYLOAD_BROKER_HPP_