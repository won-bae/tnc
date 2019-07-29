from src.core.models import MegaDetector

MODELS = {'megadetector': MegaDetector}

def build(model_config):
    graph_path = model_config['graph_path']
    model = MODELS['megadetector'](graph_path)
    return model

