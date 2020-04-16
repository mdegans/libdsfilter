# source of truth for many things
project('dsfilter', ['cuda', 'c'],
  version: '0.0.1',  # version becomes: #define DSFILTER_VERSION ... 
  meson_version: '>= 0.50.0',  # only 0.50 and above have CUDA support
)
author = 'Michael de Gans'
email = 'michael.john.degans@gmail.com'
description = 'A simple library to build CUDA DeepStream filters.'
url = 'https://github.com/mdegans/libdsfilter'
deepstream_version = '4.0.2'

# get the c compiler
cc = meson.get_compiler('c')

# get and check cuda
nvcc = meson.get_compiler('cuda')
cuda = import('unstable-cuda')
arch_flags     = cuda.nvcc_arch_flags(nvcc, 'All')
arch_readable  = cuda.nvcc_arch_readable(nvcc, 'All')
driver_version = cuda.min_driver_version(nvcc)
message('NVCC version:   ' + nvcc.version())
message('NVCC flags:     ' + ' '.join(arch_flags))
message('NVCC readable:  ' + ' '.join(arch_readable))
message('Driver version: >=' + driver_version)

subdir('config')
subdir('src')
