import torch
import torch.nn.functional as F
import torch_npu
from torch.autograd import Function

from flux_ascend._C import npu_test_ops


class TestOpsFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        x = npu_test_ops(x, y)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        pass


npu_test_ops = TestOpsFunction.apply
