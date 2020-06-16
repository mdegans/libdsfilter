#ifndef FILE_PAYLOAD_BROKER_HPP_
#define FILE_PAYLOAD_BROKER_HPP_

#pragma once

#include "ProtoPayloadFilter.hpp"
#include "queue.hpp"

#include <memory>
#include <thread>
#include <fstream>


/**
 * FileMetaBroker is a class to write metadata data to file in various
 * formats.
 */
class FileMetaBroker : public ProtoPayloadFilter {
public:
  /**
   * Magic number to prefix a binary protobuf file with (32 bit uint).
   */
  static const uint32_t PROTO_MAGIC_NUMBER = 0x5640FD6E;
  /**
   * The std::put_time format for a the csv timestamp.
   */
  static constexpr char CSV_TIME_FMT[] = "%F %T";
  /**
   * The available formats to write metadata in.
   * 
   * proto: raw protobuf in a CodedOutputStream prefixed by PROTO_MAGIC_NUMBER
   * csv: csv text format as expected by smart_distancing's frontend.
   */
  enum Format { proto, csv };

  FileMetaBroker(std::string basename, Format format = proto);
  virtual ~FileMetaBroker() = default;

  /**
   * Called by on_buffer when payload metadata is found in batch_meta's user
   * meta list.
   * 
   * @param batch_meta the deepstream bathch metadata structure
   * @param batch batch level metadata from DistanceFilter (dsdistance)
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_meta(
    NvDsBatchMeta* batch_meta, distanceproto::Batch* batch);
  /**
   * Opens the file and starts the worker thread.
   */
  virtual void start();
  /**
   * tells the worker thread to flush any data in the queue and shut down.
   * 
   * @param block blocks until queue flushed and file closed.
   */
  virtual void stop(bool block=true);
  /**
   * get the output filename
   */
  virtual std::string get_filename();

protected:
  /**
   * worker thread for writing distanceproto::Batch to a protobuf
   * CodedOutputStream
   */
  virtual void proto_worker_func();
  /**
   * worker thread for writing distanceproto::Batch to file in a format
   */
  virtual void csv_worker_func();

  // my husband complained about this->everywhere so now there are underscores
  // everywhere and to me this is more confusing. explicit "this" is like "self"
  // it's clear it refers to the instance, but code reviews are code reviews.
  std::string basepath_;
  std::thread worker_;
  my::Queue<std::unique_ptr<distanceproto::Batch>> queue_;
  Format format_;
};

#endif  // FILE_PAYLOAD_BROKER_HPP_