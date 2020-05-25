// todo: this used to run on build wiht meson. need to switch build systems
#include "TestCudaFilter.hpp"

// TODO(mdegans): use gtest

int main() {
  g_print("* creating TestCudaFilter\n");
  BaseCudaFilter* filter = nullptr;
  filter = new TestCudaFilter();
  if (filter == nullptr) {
    g_error("filter == NULL");
  }
  g_print("* deleting TestCudaFilter\n");
  delete filter;
}