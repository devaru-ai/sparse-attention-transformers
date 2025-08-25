from .transformer import Transformer
# from .reformer import Reformer  # Add other models if available
# from .rev_att import RevAtt     # Example: custom variant

def get_model(name, *args, **kwargs):
    model_map = {
        "transformer": Transformer,
        # "reformer": Reformer,
        # "rev_att": RevAtt
        # Add other models here
    }
    if name not in model_map:
        raise ValueError(f"Model '{name}' not recognized: {list(model_map.keys())}")
    return model_map[name](*args, **kwargs)
