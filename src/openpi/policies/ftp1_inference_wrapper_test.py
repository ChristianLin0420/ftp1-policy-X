from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper


class _NoiseEchoModel:
    def __init__(self) -> None:
        self.noises: list[torch.Tensor | None] = []

    def sample_actions(self, *, device, observation, noise, num_steps):
        del device, observation, num_steps
        self.noises.append(None if noise is None else noise.clone())
        return torch.zeros(1, 3, 4) if noise is None else noise


def _wrapper() -> FTP1InferenceWrapper:
    wrapper = FTP1InferenceWrapper.__new__(FTP1InferenceWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.model_config = SimpleNamespace(action_horizon=3, action_dim=4)
    wrapper.model = _NoiseEchoModel()
    wrapper.num_inference_steps = 7
    wrapper.skip_normalization = True
    wrapper.build_observation = lambda **kwargs: SimpleNamespace(state=torch.zeros(1, 1, 4))
    return wrapper


def _infer(wrapper: FTP1InferenceWrapper, generator: torch.Generator | None) -> np.ndarray:
    return wrapper.infer(
        images={},
        state=np.zeros((1, 4), dtype=np.float32),
        prompt="test",
        noise_generator=generator,
    )


def test_infer_uses_repeatable_episode_local_noise_without_touching_global_rng():
    wrapper = _wrapper()
    global_before = torch.random.get_rng_state().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    first = _infer(wrapper, generator)
    generator.manual_seed(123)
    repeated = _infer(wrapper, generator)
    generator.manual_seed(124)
    different = _infer(wrapper, generator)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
    assert all(noise is not None for noise in wrapper.model.noises)


def test_infer_preserves_legacy_model_owned_noise_when_generator_is_omitted():
    wrapper = _wrapper()

    result = _infer(wrapper, None)

    np.testing.assert_array_equal(result, np.zeros((3, 4), dtype=np.float32))
    assert wrapper.model.noises == [None]
