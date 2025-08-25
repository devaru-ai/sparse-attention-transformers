def get_model(name, *args, **kwargs):
    if name == "transformer":
        from .transformer import Transformer
        return Transformer(*args, **kwargs)
    elif name == "reformer":
        from .reformer import Reformer
        return Reformer(*args, **kwargs)
    # Add more models as needed
    else:
        raise ValueError(f"Unknown model: {name}")
