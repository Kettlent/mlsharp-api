import torch
from torch import nn
import coremltools as ct
from coremltools.models import MLModel

# 🔑 OFFICIAL BUILDER (THIS FIXES EVERYTHING)
from sharp.models import create_predictor, PredictorParams


# ------------------------------------------------
# CORE ML WRAPPER
# ------------------------------------------------
class CoreMLGaussianWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image: torch.Tensor, disparity_factor: torch.Tensor):
        # depth=None is implicit in inference
        gaussians = self.predictor(image, disparity_factor)

        # Return only tensors (Core ML requirement)
        return (
            gaussians.mean_vectors,     # [N, 3]
            gaussians.singular_values,  # [N, 3] (scales)
            gaussians.quaternions,      # [N, 4] (rotation)
            gaussians.colors,           # [N, 3]
            gaussians.opacities,        # [N, 1]
        )


# ------------------------------------------------
# LOAD MODEL (OFFICIAL WAY)
# ------------------------------------------------
CHECKPOINT = "sharp_2572gikvuh.pt"
IMG_SIZE = 1536  # matches CLI internal_shape

device = "cpu"

# Build model EXACTLY like CLI
predictor = create_predictor(PredictorParams())
state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
predictor.load_state_dict(state_dict)
predictor.eval()
predictor.to(device)

model = CoreMLGaussianWrapper(predictor).eval()

# ------------------------------------------------
# TORCHSCRIPT TRACE
# ------------------------------------------------
example_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
example_disp = torch.ones(1)

traced = torch.jit.trace(
    model,
    (example_image, example_disp),
    strict=False,
)

torch.jit.save(traced, "sharp_gaussian_coreml.pt")

# ------------------------------------------------
# CORE ML CONVERSION
# ------------------------------------------------
mlprogram = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.ImageType(
            name="image",
            shape=(1, 3, IMG_SIZE, IMG_SIZE),
            scale=1 / 255.0,
            color_layout=ct.colorlayout.RGB,
        ),
        ct.TensorType(
            name="disparity_factor",
            shape=(1,),
        ),
    ],
    outputs=[
        ct.TensorType(name="mean_vectors"),
        ct.TensorType(name="singular_values"),
        ct.TensorType(name="quaternions"),
        ct.TensorType(name="colors"),
        ct.TensorType(name="opacities"),
    ],
    compute_units=ct.ComputeUnit.ALL,
)


mlprogram.save("SharpGaussian.mlpackage")


print("✅ Saved SharpGaussian.mlpackage")
