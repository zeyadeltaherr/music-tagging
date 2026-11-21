from .dummy import build_dummy


def build_datasets(cfg):

    kind = cfg.get("kind", "dummy")

    if kind == "dummy":
        return build_dummy(cfg)

    if kind == "mtat":
        from .mtat import build_mtat
        return build_mtat(cfg)

    if kind == "mtat_ast":
        from .mtat_ast import build_mtat_ast
        return build_mtat_ast(cfg)

    raise ValueError(f"Unknown data kind '{kind}'")