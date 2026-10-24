import torch

from resnet_cifar import TrainConfig, build_model, PreActBottleneck
from resnet_cifar.model import count_parameters


def test_block_keeps_shape_when_identity():
    blk = PreActBottleneck(64, 64)
    assert blk(torch.randn(2, 64, 8, 8)).shape == (2, 64, 8, 8)
    assert blk.project is None


def test_block_downsamples_with_projection():
    blk = PreActBottleneck(64, 128, stride=2)
    assert blk(torch.randn(2, 64, 16, 16)).shape == (2, 128, 8, 8)
    assert blk.project is not None


def test_forward_logits_shape():
    model = build_model(TrainConfig())
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_works_for_other_resolutions():
    model = build_model(TrainConfig()).eval()
    assert model(torch.randn(1, 3, 48, 48)).shape == (1, 10)


def test_parameter_count_matches_docs():
    assert count_parameters(build_model(TrainConfig())) == 221_226  # see scripts/count_params.py


def test_one_optimisation_step_reduces_loss_on_tiny_batch():
    torch.manual_seed(0)
    model = build_model(TrainConfig())
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    first = loss_fn(model(x), y).item()
    for _ in range(15):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    assert loss_fn(model(x), y).item() < first
