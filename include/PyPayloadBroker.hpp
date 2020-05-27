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
 * It stores NVDS_PAYLOAD_META in GList and returns it on
 * request with get_payloads.
 */
class PyPayloadBroker : public PayloadBroker {
private:
  GList* data;
  std::mutex data_lock;
public:
  PyPayloadBroker();
  virtual ~PyPayloadBroker();
  /**
   * Called by on_buffer when a NVDS_PAYLOAD_META is found on the buffer.
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_payload(std::string* payload);
  /**
   * get a GList of string with the latest payloads as char[].
   */
  virtual GList* get_payloads();
};

#endif  // PY_PAYLOAD_BROKER_HPP_