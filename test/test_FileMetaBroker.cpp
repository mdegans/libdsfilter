#include "FileMetaBroker.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <experimental/filesystem>
#include <random>
#include <functional>

namespace fs = std::experimental::filesystem;
namespace dp = distanceproto;

namespace my {
namespace project {
namespace {

// default number of frames in a batch
const int BATCH_SIZE=8;
// number of tests batches
const int NUM_BATCHES=8;
// default wait time in between producing batches
const int BATCH_WAIT_TIME=250;
// boolean rng
// https://stackoverflow.com/questions/43329352/generating-random-boolean/43329456
auto coin_toss = std::bind(std::uniform_int_distribution<>(0,1),std::default_random_engine());

// metadata generation functions (sans BBox for now)

static void
generate_person(dp::Person* person) {
  float danger;
  // lets assume only half the time there is danger for test purposes
  if (coin_toss()) {
    double danger = rand() / (RAND_MAX + 1.0);
    person->set_danger_val(danger);
    person->set_is_danger(danger >= 1.0);
  }
}

static void
generate_frame(dp::Frame* frame) {
  int num_people = rand() / (RAND_MAX / 10);
  float sum_danger = 0.0;
  for (size_t i = 0; i < num_people; i++)
  {
    auto person = frame->add_people();
    generate_person(person);
    sum_danger += person->danger_val();
  }
  frame->set_sum_danger(sum_danger);
}

dp::Batch*
generate_batch(uint32_t batch_size = BATCH_SIZE) {
  dp::Batch* batch = new dp::Batch();
  batch->set_max_frames(batch_size);
  for (size_t i = 0; i < batch_size; i++)
  {
    generate_frame(batch->add_frames());
  }
  return batch;
}

// The fixture for testing class FileMetaBroker.
class FileMetaBrokerTest : public ::testing::Test {
 protected:
  // the test FileMetaBroker
  FileMetaBroker* fmb_;
  // the test temp path
  fs::path tmp_;
  // the test metadata temp basename
  fs::path basepath_;

 public:
  FileMetaBrokerTest() {
    tmp_ = fs::temp_directory_path() / "filemetabrokertest";
    if (fs::is_directory(tmp_)) {
      fs::remove_all(tmp_);
    }
    fs::create_directory(tmp_);
    basepath_ = tmp_ / "metadata";
    fmb_ = nullptr;
  }

  ~FileMetaBrokerTest() override {
    delete fmb_;
  }
};

// Tests construction and destruction
TEST_F(FileMetaBrokerTest, TestCreateDestroy) {
  fmb_ = new FileMetaBroker(basepath_);
}

// Tests that metadata is written out corrctly
// TODO(mdegans): write code to read proto back in
//  rn this is really only half a test
TEST_F(FileMetaBrokerTest, TestProto) {
  fmb_ = new FileMetaBroker(basepath_);
  dp::Batch* batch = nullptr;
  fmb_->start();
  for (size_t i = 0; i < NUM_BATCHES; i++)
  {
    batch = generate_batch();
    fmb_->on_batch_meta(nullptr, batch);
    delete batch;
  }
  fmb_->stop();
}

TEST_F(FileMetaBrokerTest, TestCsv) {
  fmb_ = new FileMetaBroker(basepath_, FileMetaBroker::Format::csv);
  dp::Batch* batch = nullptr;
  fmb_->start();
  for (size_t i = 0; i < NUM_BATCHES; i++)
  {
    batch = generate_batch();
    fmb_->on_batch_meta(nullptr, batch);
    delete batch;
  }
  fmb_->stop();
}

}  // namespace
}  // namespace project
}  // namespace my

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}