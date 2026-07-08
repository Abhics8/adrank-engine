"""Model/artifact store. The serving layer only ever loads from a versioned
run directory written by training -- it never trains or fits anything
itself, so the serving process stays stateless and restart-safe."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.calibrate import Calibrator
from src.models.gbdt import GBDTRanker
from src.retrieval.index import AdIndex
from src.retrieval.two_tower import TwoTowerModel


@dataclass
class Artifacts:
    gbdt: GBDTRanker
    calibrator: Calibrator
    index: AdIndex
    two_tower: TwoTowerModel
    ad_feature_lookup: dict
    metrics: dict


def save_artifacts(run_dir: str, gbdt, calibrator, index, two_tower, ad_feature_lookup, metrics):
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "gbdt.pkl", "wb") as f:
        pickle.dump(gbdt, f)
    with open(d / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(d / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(d / "two_tower.pkl", "wb") as f:
        pickle.dump(two_tower, f)
    with open(d / "ad_feature_lookup.pkl", "wb") as f:
        pickle.dump(ad_feature_lookup, f)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_artifacts(run_dir: str) -> Artifacts:
    d = Path(run_dir)
    with open(d / "gbdt.pkl", "rb") as f:
        gbdt = pickle.load(f)
    with open(d / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(d / "index.pkl", "rb") as f:
        index = pickle.load(f)
    with open(d / "two_tower.pkl", "rb") as f:
        two_tower = pickle.load(f)
    with open(d / "ad_feature_lookup.pkl", "rb") as f:
        ad_feature_lookup = pickle.load(f)
    metrics = json.loads((d / "metrics.json").read_text())
    return Artifacts(gbdt, calibrator, index, two_tower, ad_feature_lookup, metrics)
