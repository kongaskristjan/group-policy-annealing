import torch

from lib.loss_value_function import discount_cumsum


def test_discount_cumsum():
    # Test for 1D tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    result = discount_cumsum(x, discount_factor=0.5, dim=0)
    expected = torch.tensor([2.75, 3.5, 3.0], dtype=torch.float64)
    assert torch.allclose(result, expected), f"1D test failed: Expected {expected}, got {result}"

    # Test for 2D tensor along dim=0
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = discount_cumsum(x, discount_factor=0.5, dim=0)
    expected = torch.tensor([[3.75, 5.5], [5.5, 7.0], [5.0, 6.0]], dtype=torch.float64)
    assert torch.allclose(result, expected), f"2D dim=0 test failed: Expected {expected}, got {result}"

    # Test for 2D tensor along dim=1
    result = discount_cumsum(x, discount_factor=0.5, dim=1)
    expected = torch.tensor([[2.0, 2.0], [5.0, 4.0], [8.0, 6.0]], dtype=torch.float64)
    assert torch.allclose(result, expected), f"2D dim=1 test failed: Expected {expected}, got {result}"

    # Test for 1D tensor with discount_factor=0.9
    x = torch.tensor([1.0, 1.0, 1.0])
    result = discount_cumsum(x, discount_factor=0.9, dim=0)
    expected = torch.tensor([1 + 0.9 + 0.9**2, 1 + 0.9, 1.0], dtype=torch.float64)
    assert torch.allclose(result, expected), f"1D discount=0.9 test failed: Expected {expected}, got {result}"
