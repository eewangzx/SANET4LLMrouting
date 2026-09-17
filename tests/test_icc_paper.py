import json

import pytest

from edge_msd.icc_paper import (
    RESOURCE_RANGES,
    generate_dataset,
    load_scenario,
    save_dataset,
)
from edge_msd.network import Network


def test_paper_units_and_resource_order(tmp_path):
    path = tmp_path / "scenario.json"
    data = generate_dataset(2026)
    save_dataset(data, path)
    scenario = load_scenario(path)
    for service in scenario.services.values():
        source_bounds = RESOURCE_RANGES[service.kind]
        for value, index in zip(service.resources, (0, 2, 1, 3)):
            assert source_bounds[index][0] <= value <= source_bounds[index][1]
        if service.kind == "core":
            raw = data["sampled_service_quantities"][service.name]
            assert service.processing_ms == pytest.approx(raw["work_mb"] / raw["rate_mb_per_ms"])
    for a, b, gbps, propagation in scenario.links:
        assert 0.8 <= gbps <= 8
        assert propagation == 0
        assert Network(scenario).delay(a, b, 1) <= 8 / gbps + 1e-8
    for user in scenario.users:
        assert all(150 <= rate <= 1500 for rate in user.rates_per_second.values())


def test_figure_dags_and_reproducible_load_scaling(tmp_path):
    data = generate_dataset(2026)
    assert data == generate_dataset(2026)
    assert data != generate_dataset(2027)
    path = tmp_path / "scenario.json"
    save_dataset(data, path)
    base, high = load_scenario(path), load_scenario(path, 2)
    assert (len(base.core), len(base.light), len(base.tasks), len(base.nodes)) == (6, 9, 4, 10)
    assert set(base.tasks["cross_modal_understanding"].dependencies["projection_c"]) == {
        "text_encoder",
        "image_encoder",
    }
    assert set(base.tasks["multimodal_driving"].dependencies["projection_c"]) == {
        "image_encoder",
        "compression",
    }
    assert base.links == high.links
    for u, v in zip(base.users, high.users):
        assert all(v.rates_per_second[t] == 2 * r for t, r in u.rates_per_second.items())
    data["resource_order"] = ["CPU", "RAM", "GPU", "VRAM"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resource order"):
        load_scenario(path)
