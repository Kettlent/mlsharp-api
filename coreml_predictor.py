import torch
from torch import nn

class CoreMLGaussianPredictor(nn.Module):
    """
    Core ML–exportable wrapper.
    Converts Gaussians3D → raw tensors.
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image: torch.Tensor, disparity_factor: torch.Tensor):
        # Inference mode: depth = None
        gaussians = self.predictor(
            image=image,
            disparity_factor=disparity_factor,
            depth=None,
        )

        return (
            gaussians.means,        # [N, 3]
            gaussians.covariances,  # [N, 6]
            gaussians.colors,       # [N, 3]
            gaussians.opacity,      # [N, 1]
        )
