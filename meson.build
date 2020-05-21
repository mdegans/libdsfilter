# source of truth for many things
project('dsfilter', ['cuda', 'cpp'],
  # version becomes: #define DSFILTER_VERSION ... 
  version: run_command(
    'head', '-n', '1', files('VERSION'),
  ).stdout().strip(),
  # only 0.50 and above have CUDA support
  meson_version: '>= 0.54.0',
  license : 'LGPL',
)
author = 'Michael de Gans'
email = 'michael.john.degans@gmail.com'
description = 'A simple library to build CUDA DeepStream filters.'
url = 'https://github.com/mdegans/libdsfilter'
deepstream_version = '5.0'

# get the c compiler
cc = meson.get_compiler('cpp')

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
