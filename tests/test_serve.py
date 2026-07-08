"""API contract tests using a tiny in-process trained model, so they don't
depend on a pre-existing models/ directory."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import yaml

    from src.train import main as train_main

    run_dir = tmp_path_factory.mktemp("run")
    config = {
        "seed": 7,
        "data": {"days": 5, "rows_per_day": 800},
        "two_tower": {"enabled": True, "epochs": 1, "retrieve_k": 20},
    }
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.dump(config))

    import sys

    sys.argv = ["train.py", "--config", str(config_path), "--run-id", "test"]
    os.chdir(run_dir)
    train_main()

    os.environ["ADRANK_RUN_DIR"] = str(run_dir / "models" / "test")
    from src.serve.app import app

    # TestClient only fires FastAPI's startup event (which loads artifacts
    # into the module-level _artifacts global) when used as a context
    # manager -- plain instantiation silently skips lifespan handling.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_rank_returns_sorted_by_expected_value(client):
    resp = client.post(
        "/rank",
        json={
            "user_id": 1,
            "device_id": 2,
            "hour": 14,
            "candidate_bids": {"1": 2.0, "5": 1.0, "10": 3.0},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    values = [r["expected_value"] for r in body["ranked"]]
    assert values == sorted(values, reverse=True)
    assert "retrieve_ms" in body["latency_ms"]
