# basic build configuration
src = [
  'BaseCudaFilter.cu',
  'TestCudaFilter.cu',
  'BaseFilter.cpp',
  'DistanceFilter.cpp',
  'meta.pb.cc',
]
includes = [
  include_directories('../include'),
  include_directories('../config'),
]
headers_to_install = [
  '../include/BaseCudaFilter.hpp',
  '../include/TestCudaFilter.hpp',
  '../include/BaseFilter.hpp',
  '../include/DistanceFilter.hpp',
]

# deepstream stuff, becuase deepstream.pc file is not in PKG_CONFIG_PATH
# and include path is incorrect.
nvds_libdir = '/opt/nvidia/deepstream/deepstream-5.0/lib'
nvds_incdir = '/opt/nvidia/deepstream/deepstream-5.0/sources/includes'
nvds_link_args = ['-L'+nvds_libdir, '-lnvds_meta', '-lnvdsgst_meta']
nvds_pkgconfig_deps = [
  dependency('protobuf', version: '>=3.0.0')
  dependency('glib-2.0', version: '>=2.50'),
  dependency('gobject-2.0', version: '>=2.0'),
  dependency('gstreamer-1.0', version: '>=1.0'),
]
nvds_dep = declare_dependency(
  version: deepstream_version,
  dependencies: [
    dependency('cuda', version: '>=10'),
    nvds_pkgconfig_deps,
  ],
  link_args: nvds_link_args,
  include_directories: include_directories(nvds_incdir),
)

libdsfilter_name = meson.project_name() + '-' + meson.project_version()

# shared library target
libdsfilter = shared_library(libdsfilter_name, src,
  dependencies: nvds_dep,
  include_directories: includes,
  install: true,
)

# a dep so this can be included as a subproject
libdsfilter_dep = declare_dependency(
  version: meson.project_version(),
  dependencies: nvds_dep,
  include_directories: [includes, include_directories(nvds_incdir)],
  link_args: nvds_link_args,
  link_with: libdsfilter,
  # variables: dsfilter_config,  # must be a dict, so skipping for now
)

# install headers to $PREFIX/include/dsfilter
install_headers(headers_to_install, subdir: meson.project_name())

subdir('pkg-config')
subdir('test')
