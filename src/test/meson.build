# unit tests
# TODO(mdegans): write more tests
tests = [
  {
    'description': 'Test CUDA inteface creation and destruction',
    'filename': 'test_cu_create_destroy',
    'sources': ['test_cu_create_destroy.cu'],
  },
]
foreach t: tests
  exe = executable(t['filename'], t['sources'],
    dependencies: nvds_dep,
    link_with: libdsfilter,
    include_directories: includes,
  )
  test(t['description'], exe, is_parallel: false)
endforeach
