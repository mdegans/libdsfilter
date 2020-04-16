#include "TestCudaFilter.hpp"

// TODO(mdegans): use gtest

int main() {
    g_print("* creating TestCudaFilter\n");
    BaseCudaFilter * filter = NULL;
    filter = new TestCudaFilter();
    if (filter == NULL) {
        g_error("filter == NULL");
    }
    g_print("* deleting TestCudaFilter\n");
    delete filter;
}