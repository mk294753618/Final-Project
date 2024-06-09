def parse_data_config(path):
    options = dict()
    options['gpu'] = '0'
    options['num_worker'] = '0'
    file = open(path, 'r')
    lines = file.read().split('\n')
    for line in lines:
        line = line.strip()
        if line == ' ' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [line.rstrip().lstrip() for line in lines]
    model_defs = []
    for line in lines:
        if line.startswith('#'):
            continue
        if line:
            if line.startswith('['):
                model_defs.append({})
                model_defs[-1]['type'] = line[1:-1].rstrip()
                if model_defs[-1]['type'] == 'convolutional':
                    model_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split('=')
                model_defs[-1][key.rstrip()] = value.strip()
    return model_defs


