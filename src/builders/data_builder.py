from src.utils.util import load_json, load_image_paths, log

def build(data_config):
    image_dir = data_config['image_dir']
    json_file = data_config.get('json_file', None)
    batch_size = data_config.get('batch_size', 1)

    def _batch(image_paths, labels, total_num, batch_size):
        for idx in range(0, total_num, batch_size):
            log.infov('Processing batch : {}/{}'.format(
                int(idx/batch_size)+1, int(total_num/batch_size)))
            yield (image_paths[idx:min(idx + batch_size, total_num)],
                   labels[idx:min(idx + batch_size, total_num)])

    if json_file:
      image_paths, labels = load_json(image_dir, json_file)
    else:
      recursive = data_config.get('recursive', False)
      image_paths, labels = load_image_paths(image_dir, recursive)
    total_num = len(labels)
    return _batch(image_paths, labels, total_num, batch_size)
