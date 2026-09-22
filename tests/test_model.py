import torch

from airbourne_classifier.constants import FIXED_FRAMES, LABELS, N_MELS
from airbourne_classifier.model import SmallAudioCNN


def test_forward_output_shape():
    model = SmallAudioCNN()
    batch = torch.randn(4, 1, N_MELS, FIXED_FRAMES)
    out = model(batch)
    assert out.shape == (4, len(LABELS))


def test_param_count_is_light():
    model = SmallAudioCNN()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 500_000
