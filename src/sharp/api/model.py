import logging
import torch

from sharp.models import PredictorParams, create_predictor

LOGGER = logging.getLogger(__name__)

_predictor = None
_device = None


def load_model(checkpoint_path: str | None = None):
    global _predictor, _device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SHARP API")

    _device = torch.device("cuda")

    LOGGER.info("Loading SHARP model on CUDA")

    if checkpoint_path is None:
        from sharp.cli.predict import DEFAULT_MODEL_URL
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True
        )
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(_device)

    _predictor = predictor
    LOGGER.info("Model loaded successfully")


def get_model():
    if _predictor is None:
        raise RuntimeError("Model not loaded")
    return _predictor, _device
