from pandas import DataFrame
import os
import yaml

# Read the configuration file for this run
with open(os.path.join('../config', f'{config_file_name}.yaml'), 'r') as stream:
    config_params = yaml.safe_load(stream.read())

# Read the yaml config file
with open(os.path.join('../config', f'{default_config_file_name}.yaml'), 'r') as stream:
    default_params = yaml.safe_load(stream.read())

# Create metadata from config inputs
param_combos_df = DataFrame({'uid': ['']})
for param in sorted(set(config_params.keys()).union(set(default_params.keys()))):
    param_df = DataFrame()

    try:
        params_val = config_params[param]
        if params_val is None:
            params_val = default_params[param]
    except KeyError:
        params_val = default_params[param]

    if param == 'fw':
        param_df['fw_start'] = params_val['start']
        param_df['fw_end'] = params_val['end']
    else:
        param_df[param] = params_val
    param_df['uid'] = ''
    param_combos_df = param_combos_df.merge(param_df, how='outer', on='uid')

# Populate uid from other metadata
meta_cols = list(param_combos_df.columns)
meta_cols.remove('uid')
param_combos_df['uid'] = config_file_name
for col in meta_cols:
    col_initials = ''.join([x[0] for x in col.split('_')])
    param_combos_df['uid'] = param_combos_df['uid'] + "_" + col_initials + param_combos_df[col].astype(str)
