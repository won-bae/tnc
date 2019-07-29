import argparse
from src.engine import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default.yml',
                        help='config name')
    parser.add_argument('--tag', default='',
                        help='tag to discern evaluation instances')
    args = parser.parse_args()

    engine = Engine(config_name=args.config, tag=args.tag)
    engine.evaluate()
