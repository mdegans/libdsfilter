#include "TestCudaFilter.hpp"

bool TestCudaFilter::on_object(NvDsFrameMeta * f_meta, NvDsObjectMeta * o_meta, NvBufSurfaceParams * frame) {
    GST_DEBUG("TestCudaFilter:on_object:got frame %d", f_meta->frame_num);
    GST_DEBUG("TestCudaFilter:on_object:surface color format: %d", frame->colorFormat);
    GST_DEBUG("TestCudaFilter:on_object:got object: %s with confidence %.2f",
        o_meta->obj_label, o_meta->confidence);
    return true;
}