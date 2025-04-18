from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F

## Define gradient reversal function
class GradientReverseFunction(Function):
    """
                Rewrite the customized gradient calculation method
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverse(nn.Module):
    def __init__(self):
        super(GradientReverse, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
