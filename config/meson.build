# add HAVE_CONFIG define to cuda and c compilers
add_project_arguments('-DHAVE_CONFIG_H', language: ['cuda', 'c'])

# get a configuration data object
dsfilter_config = configuration_data()

# set properties on configuration data object
dsfilter_config.set_quoted('DSFILTER_VERSION', meson.project_version())
dsfilter_config.set('TEGRA', target_machine.cpu_family() == 'aarch64')
dsfilter_config.set('VERBOSE', true)
dsfilter_config.set('VERY_VERBOSE', false)
dsfilter_config.set_quoted('DSFILTER_AUTHOR', author)
dsfilter_config.set_quoted('DSFILTER_EMAIL', email)
dsfilter_config.set_quoted('DSFILTER_DESCRIPTION', description)
dsfilter_config.set_quoted('DSFILTER_URL', url)

# output the configuration file using the configuration data object
configure_file(output : 'dsfilter_config_meson.h', configuration : dsfilter_config)
