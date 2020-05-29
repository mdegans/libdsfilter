#ifndef PY_PAYLOAD_BROKER_HPP_
#define PY_PAYLOAD_BROKER_HPP_

#pragma once

#include "PayloadBroker.hpp"

#include <thread>
#include <mutex>
#include <queue>

/**
 * PyPayloadBroker is a class designed to be used from other languages.
 * (not necessarily Python, but i'm feeling too lazy to rename at the moment)
 * 
 * It stores NVDS_PAYLOAD_META in gchararray and returns it on
 * request with get_payloads.
 */
class PyPayloadBroker : public PayloadBroker {
private:
  gchararray data;
  std::mutex data_lock;
public:
  PyPayloadBroker();
  virtual ~PyPayloadBroker() = default;
  /**
   * Called by on_buffer when a NVDS_PAYLOAD_META is found on the buffer.
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_payload(std::string* payload);
  /**
   * get a gchararray with the latest serialized batch.
   */
  virtual gchararray get_payload();
};

#endif  // PY_PAYLOAD_BROKER_HPP_