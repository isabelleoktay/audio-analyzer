import numpy as np

def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_builtin_types(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj