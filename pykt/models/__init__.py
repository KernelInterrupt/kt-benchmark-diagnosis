from .evaluate_model import evaluate, evaluate_question, evaluate_splitpred_question, effective_fusion


def train_model(*args, **kwargs):
    from .train_model import train_model as _train_model

    return _train_model(*args, **kwargs)


def train_model4promptkt(*args, **kwargs):
    from .train_model4promptkt import train_model4promptkt as _train_model4promptkt

    return _train_model4promptkt(*args, **kwargs)


def init_model(*args, **kwargs):
    from .init_model import init_model as _init_model

    return _init_model(*args, **kwargs)


def load_model(*args, **kwargs):
    from .init_model import load_model as _load_model

    return _load_model(*args, **kwargs)


def init_model4promptkt(*args, **kwargs):
    from .init_model4promptkt import init_model4promptkt as _init_model4promptkt

    return _init_model4promptkt(*args, **kwargs)


def load_model4promptkt(*args, **kwargs):
    from .init_model4promptkt import load_model4promptkt as _load_model4promptkt

    return _load_model4promptkt(*args, **kwargs)


def lpkt_evaluate_multi_ahead(*args, **kwargs):
    from .lpkt_utils import lpkt_evaluate_multi_ahead as _lpkt_evaluate_multi_ahead

    return _lpkt_evaluate_multi_ahead(*args, **kwargs)
