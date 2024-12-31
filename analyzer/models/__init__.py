import importlib

def create_model(version, segment_size):
    print(f"Trying to import model v{version}")
    module = importlib.import_module(f'analyzer.models.v{version}')
    return module.create_model(segment_size)
