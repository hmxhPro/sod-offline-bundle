"""
tests/test_misc.py
------------------
Covers M-2 (deploy gate) and the M-1 val_is_train API plumbing — pure logic,
no DB.
"""

from __future__ import annotations

import types
from datetime import datetime, timezone


# ── M-2: deployability gate ─────────────────────────────────────────────────

def test_is_deployable_default_floor():
    from app.services.training_manager import TrainingManager
    assert TrainingManager._is_deployable(None) is False     # no metric
    assert TrainingManager._is_deployable(0.0) is False      # all-zero broken run
    assert TrainingManager._is_deployable(0.5) is True


def test_is_deployable_custom_floor(monkeypatch):
    import app.services.training_manager as tmgr
    monkeypatch.setattr(tmgr.settings, "MIN_DEPLOYABLE_MAP50", 0.3)
    assert tmgr.TrainingManager._is_deployable(0.2) is False
    assert tmgr.TrainingManager._is_deployable(0.4) is True


# ── M-1: val_is_train surfaced from params (jobs) / metrics (models) ────────

def _job_rec(params):
    return types.SimpleNamespace(
        id="j1", category_id="c1", model_name="m", status="finished",
        progress=1.0, current_epoch=10, total_epochs=10,
        dataset_yaml="/d.yaml", base_model="/b.pt", metric_map50=0.5,
        metric_map50_95=0.3, metrics={}, best_pt_path="/best.pt", error=None,
        val_is_train=False, created_at=datetime.now(timezone.utc),
        started_at=None, finished_at=None, params=params,
    )


def _model_rec(metrics):
    return types.SimpleNamespace(
        id="m1", name="Cat", version=1, category_id="c1", training_job_id="j1",
        weights_path="/best.pt", base_model="/b.pt", class_names={"0": "Cat"},
        dataset_yaml="/d.yaml", num_images=10, metrics=metrics, val_is_train=False,
        trained_started_at=None, trained_finished_at=None,
        created_at=datetime.now(timezone.utc),
    )


def test_job_item_surfaces_val_is_train():
    from app.api.training import _job_item
    assert _job_item(_job_rec({"val_is_train": True})).val_is_train is True
    assert _job_item(_job_rec({"val_is_train": False})).val_is_train is False
    assert _job_item(_job_rec(None)).val_is_train is False        # legacy row → default


def test_model_item_surfaces_val_is_train():
    from app.api.models import _model_item
    assert _model_item(_model_rec({"val_is_train": True})).val_is_train is True
    assert _model_item(_model_rec({"mAP50": 0.5})).val_is_train is False  # legacy metrics
    assert _model_item(_model_rec(None)).val_is_train is False
