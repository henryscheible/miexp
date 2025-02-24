import pytest
import torch
from plotly.graph_objs import Figure

from miexp.visualizations.param_vis import visualize_state_dict_at_epoch


@pytest.fixture
def sample_state_dict():
    return {
        "attention_head.w_q.weight": torch.randn(3, 3, 3),
        "attention_head.w_k.weight": torch.randn(3, 3, 3),
        "attention_head.w_v.weight": torch.randn(3, 3, 3),
        "attention_head.w_o.weight": torch.randn(3, 3, 3),
    }


def test_visualize_state_dict_at_epoch(sample_state_dict):
    fig = visualize_state_dict_at_epoch(sample_state_dict)

    assert isinstance(fig, Figure)
    assert (
        len(fig.data) == len(sample_state_dict) + 2  # type: ignore
    )  # Original matrices + Effective QKV + Effective QKVO
    assert fig.layout.height == 300 * len(sample_state_dict) / 3  # type: ignore
    assert not fig.layout.showlegend  # type: ignore
