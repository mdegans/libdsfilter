/* FileMetaBroker.cpp
 *
 * Copyright 2020 Michael de Gans <47511965+mdegans@users.noreply.github.com>
 *
 * 66E67F6ADF56899B2AA37EF8BF1F2B9DFBB1D82E66BD48C05D8A73074A7D2B75
 * EB8AA44E3ACF111885E4F84D27DC01BB3BD8B322A9E8D7287AD20A6F6CD5CB1F
 *
 * This file is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

// TODO(mdegans):
//  - rename these classes and the base class to MetaBroker?
//  - 

#include "FileMetaBroker.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <thread>

namespace dp = distanceproto;

GST_DEBUG_CATEGORY_STATIC(filemetabroker);

FileMetaBroker::FileMetaBroker(std::string basepath, Format format) :
  basepath_(basepath),
  format_(format),
  queue_()
  {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    GST_DEBUG_CATEGORY_INIT(
      filemetabroker, "filemetabroker", 0,
      "FileMetaBroker debug category.");
    GST_DEBUG("%s end", __func__);
  }

std::string
FileMetaBroker::get_filename(){
  switch (format_){
    case csv:
      return basepath_ + ".csv";
    case proto:
      return basepath_ + ".coded";
  }
}

void
FileMetaBroker::proto_worker_func() {
  GST_DEBUG("%s start", __func__);
  GST_DEBUG("opening %s", get_filename().c_str());
// https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.io.coded_stream
  int fd = open(get_filename().c_str(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    GST_ERROR("could not open for output: %s", get_filename().c_str());
    return;
  }
  // Google's fancy, potentially zero-copy, ostream
  auto raw = new google::protobuf::io::FileOutputStream(fd);
  // the coded output stream
  auto coded = new google::protobuf::io::CodedOutputStream(raw);
  coded->WriteLittleEndian32(PROTO_MAGIC_NUMBER);
  // wait for the first batch
  auto batch = std::move(queue_.get());
  while (batch) {
    if (!batch->SerializeToCodedStream(coded)) {
      GST_ERROR("failed to write to %s", get_filename().c_str());
      break;
    };
    // get the next batch (or nullptr)
    batch = std::move(queue_.get());
  }
  delete coded;
  delete raw;
  if (close(fd) == EIO) {
    GST_ERROR("I/O error closing %s", get_filename().c_str());
  }
}

/**
 * Write a dp::Frame to a csv output stream in the format expected by
 * neuralet/smart_distancing:
 * 
 * Timestamp,DetectedObjects,ViolatingObjects,EnvironmentScore
 */
static void
frame_to_csv(const dp::Frame& frame, std::ostream& out) {
  // so we'll get the danger score by dividing the sum danger by the number
  // of people. This has the potential to be greater than 1 in extreme cases.
  float score = 0.0f;
  if (frame.people_size() > 0) {
    score = frame.sum_danger() / frame.people_size();
  }
  // number of violating people
  int violating = 0;
  for (size_t i = 0; i < frame.people_size(); i++)
  {
    const dp::Person& person = frame.people(i);
    if (person.is_danger()) violating++;
  }
  auto time = std::time(nullptr);
  out << std::put_time(std::localtime(&time), FileMetaBroker::CSV_TIME_FMT)
    << "," << frame.people_size() << "," << violating << "," << score << "\n";
}

void
FileMetaBroker::csv_worker_func() {
  GST_DEBUG("%s start", __func__);
  // output file opened for appending
  std::ofstream out(get_filename().c_str(), std::ios::out | std::ios::app);
  if (!out.is_open()) {
    GST_ERROR("failed to open %s", get_filename().c_str());
    return;
  }
  // set the number of digits for floats
  out << std::setprecision(3);
  // wait for the first batch
  auto batch = std::move(queue_.get());
  while (batch) {
    for (size_t i = 0; i < batch->frames_size(); i++)
    {
      frame_to_csv(batch->frames(i), out);
    }
    // flush after each batch for testing (remove this).
    out.flush();
    // get the next batch (or nullptr)
    batch = std::move(queue_.get());
  };
  out.close();
}

bool
FileMetaBroker::on_batch_meta(NvDsBatchMeta* batch_meta, dp::Batch* batch) {
  GST_LOG("%s start", __func__);
  // make a copy of the batch and stick it in a unique_ptr
  auto batch_copy = std::unique_ptr<dp::Batch>(new dp::Batch(*batch));
  // move the unique_ptr into the queue
  this->queue_.put(std::move(batch_copy));
}

void
FileMetaBroker::start() {
  GST_DEBUG("%s start", __func__);
  switch (format_){
    case csv:
      GST_DEBUG("spawning csv worker thread");
      worker_ = std::thread(&FileMetaBroker::csv_worker_func, this);
      break;
    case proto:
      GST_DEBUG("spawning proto worker thread");
      worker_ = std::thread(&FileMetaBroker::proto_worker_func, this);
  }
}

void
FileMetaBroker::stop(bool block) {
  GST_DEBUG("%s start", __func__);
  queue_.flush();
  if (block && worker_.joinable()){
    GST_DEBUG("%s joining", __func__);
    worker_.join();
  }
  GST_DEBUG("%s end", __func__);
}
