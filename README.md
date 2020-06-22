# libdsfilter

Is a simple library to aid in the creation of DeepStream filter plugins. The various classes are intended for use with standard GStreamer boilerplate element templates. Example usage can be found in gst-cuda-plugin.

BaseFilter - a base class for all filters.
-> BaseCudaFilter - a base class for cuda filters (it has a cuda stream).

For the moment, please see the headers for documentation. HTML documentation is planned.