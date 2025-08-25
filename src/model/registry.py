def get_model(name, *args, **kwargs):
    if name == "transformer":
        from .transformer import Transformer
        return Transformer(*args, **kwargs)
    elif name == "reformer":
        from .reformer import Reformer
        return Reformer(*args, **kwargs)
    elif name == "reformer_pp":
        from .reformer_pp import ReformerPP
        return ReformerPP(*args, **kwargs)
    # Add more models as needed
    else:
        raise ValueError(f"Unknown model: {name}")
