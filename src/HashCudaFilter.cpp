#include "HashCudaFilter.hpp"


__global__
void calc_hash(char* data, Hash& hash) {

}


bool HashCudaFilter::on_frame(NvBufSurface* surf, NvDsFrameMeta* frame_meta) {
  // NvBufSurfaceParams* frame = &surf->surfaceList[frame_meta->batch_id];
  GST_WARNING("currently, the hashfilter does nothing");
  // cudaMallocManaged(char, sizeof(Hash));
  return true;
}