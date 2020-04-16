# generate a .pc to work around
# DeepStream's lack of .pc or .cmake
pkg = import('pkgconfig')
pkg.generate(libdsfilter,
  name: meson.project_name(),
  version: meson.project_version(),
  description: description,
  filebase: meson.project_name(),
  libraries: nvds_link_args,
  requires: nvds_pkgconfig_deps,
  url: url,
  extra_cflags: ['-I'+nvds_incdir]
)