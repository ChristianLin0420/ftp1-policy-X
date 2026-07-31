import pytest

import scripts.zarr_compute_norm_stats as compute_norm_stats


def test_main_finalizes_ftp1_config_before_creating_data(monkeypatch):
    events = []

    class FakeConfig:
        assets_dirs = "assets"
        model = object()

        @property
        def data(self):
            events.append("data")
            return self

        def finalize_config(self):
            events.append("finalize")

        def create(self, _assets_dirs, _model):
            events.append("create")
            raise RuntimeError("stop after data creation")

    monkeypatch.setattr(compute_norm_stats._config, "FTP1TrainConfig", FakeConfig)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="stop after data creation"):
        compute_norm_stats.main(FakeConfig())

    assert events == ["finalize", "data", "create"]
