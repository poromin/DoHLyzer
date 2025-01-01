import importlib

def create_model(version, segment_size):
    if version in [1, 2]:
        print("Classification of DoH vs Non-DoH.")
    elif version in [3, 4]:
        print("Classification of Benign DoH vs Malicious DoH.")
    else:
        print("Invalid model version.")
    module = importlib.import_module(f'analyzer.models.v{version}')
    return module.create_model(segment_size)
