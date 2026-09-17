"""Wire correctness, causality and trainability of coupled ICC telemetry codecs."""

import numpy as np
import pytest
import torch

from edge_msd.icc_coupled.codecs import KINDS, CoupledCodec, load_codec, train_codecs


@pytest.mark.parametrize("kind,words", tuple(zip(KINDS, (128, 16, 4, 4, 4, 4), strict=True)))
def test_codec_wire_roundtrip_and_causal_receiver(kind, words, tmp_path):
    torch.manual_seed(3)
    model = CoupledCodec(kind, horizon=7).eval()
    rng = np.random.default_rng(4)
    history = rng.uniform(0.2, 1.2, (8, 16)).astype(np.float32)
    original = history.copy()
    payload = model.encode_wire(history)
    assert payload.shape == (words,)
    assert payload.dtype == np.float32
    assert payload.nbytes == model.payload_bytes == 4 * words
    wire_prediction = model.forecast_wire(payload)
    with torch.no_grad():
        direct = model(torch.from_numpy(original)[None])[0].numpy()
    np.testing.assert_allclose(wire_prediction, direct, rtol=1e-6, atol=1e-6)
    assert wire_prediction.shape == (7, 14)
    # Receiver cannot access changed source state, labels, or a cached window.
    history[:] = 100
    _ = model.encode_wire(rng.uniform(0.2, 1.2, (8, 16)).astype(np.float32))
    np.testing.assert_array_equal(model.forecast_wire(payload), wire_prediction)
    model.save(tmp_path / f"{kind}.pt")
    receiver = load_codec(tmp_path / f"{kind}.pt")
    assert receiver.encoding_hash() == model.encoding_hash()
    np.testing.assert_array_equal(receiver.forecast_wire(payload), wire_prediction)


def test_importance_mask_validation_and_prediction_gradient():
    torch.manual_seed(13)
    model = CoupledCodec("importance", horizon=3)
    x = torch.rand(5, 8, 16)
    z, scores, mask = model.representation(x)
    assert (mask.detach().sum(-1) == 3).all()
    assert (z.detach().count_nonzero(-1) == 3).all()
    model(x).sum().backward()
    assert model.importance.weight.grad.abs().sum() > 0
    assert model.encoder[1].weight.grad.abs().sum() > 0
    for invalid in ([0, 1, 2, 3], [255, 1, 2, 3], [7.5, 1, 2, 3], [7, 1, 2],
                    [7, 1, 2, np.nan]):
        with pytest.raises(ValueError):
            model.forecast_wire(invalid)


def test_stats4_current_alignment_and_slopes():
    model = CoupledCodec("stats4", horizon=4)
    x = np.ones((8, 16), dtype=np.float32)
    x[:, :9] = 0.4 + np.arange(8)[:, None] * 0.02
    x[:, 9:14] = 1.1 - np.arange(8)[:, None] * 0.01
    payload = model.encode_wire(x)
    np.testing.assert_allclose(payload, [0.54, 1.03, 0.02, -0.01], atol=1e-6)
    prediction = model.forecast_wire(payload)
    np.testing.assert_allclose(prediction[:, 0], [0.54, 0.56, 0.58, 0.60], atol=1e-6)
    np.testing.assert_allclose(prediction[:, 9], [1.03, 1.02, 1.01, 1.00], atol=1e-6)


def test_small_training_freezes_ae_and_infers_horizon(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 1.1, (20, 8, 16)).astype(np.float32)
    y = np.repeat(x[:, -1:, :14], 4, axis=1)
    models = train_codecs(x[:16], y[:16], x[16:], y[16:], tmp_path,
                          epochs=1, batch_size=8, kinds=("ae4", "importance", "stats4"))
    assert models["ae4"].horizon == 4
    assert all(not p.requires_grad for p in models["ae4"].encoder.parameters())
    assert (tmp_path / "metrics.json").exists()
    for model in models.values():
        assert np.isfinite(model.forecast_wire(model.encode_wire(x[0]))).all()
