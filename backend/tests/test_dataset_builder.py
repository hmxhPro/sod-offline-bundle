"""
tests/test_dataset_builder.py
-----------------------------
Covers M-3: the train/val split is seeded (reproducible) and split.json is
frozen alongside the dataset; val_is_train triggers when the holdout is too small.
"""

from __future__ import annotations

import json

from app.services import dataset_builder as db


def _make_dataset(tmp_path, monkeypatch, n: int):
    monkeypatch.setattr(db.settings, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(db.settings, "ANNOTATIONS_DIR", tmp_path / "ann")
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (tmp_path / "ann" / "cat").mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(n):
        p = raw / f"img{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff")           # bytes only; finalize just copies
        (tmp_path / "ann" / "cat" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2")
        images.append({"id": f"img{i}", "stored_path": str(p)})
    return images


def _split(tmp_path, job: str) -> dict:
    return json.loads(
        (tmp_path / "datasets" / "cat" / "yolo" / job / "split.json").read_text()
    )


def test_split_is_seeded_and_reproducible(tmp_path, monkeypatch):
    images = _make_dataset(tmp_path, monkeypatch, n=30)   # 24/6 → real holdout
    r1 = db.finalize("cat", "Cat", "job1", list(images))
    r2 = db.finalize("cat", "Cat", "job2", list(images))
    s1, s2 = _split(tmp_path, "job1"), _split(tmp_path, "job2")
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["seed"] == db.settings.TRAIN_SPLIT_SEED
    assert r1["val_is_train"] is False and r2["val_is_train"] is False
    assert set(s1["train"]).isdisjoint(s1["val"])         # real holdout, no overlap


def test_val_mirrors_train_when_too_few(tmp_path, monkeypatch):
    images = _make_dataset(tmp_path, monkeypatch, n=6)     # 6 → val(1) < MIN_VAL_IMAGES
    r = db.finalize("cat", "Cat", "jobX", list(images))
    assert r["val_is_train"] is True
    s = _split(tmp_path, "jobX")
    assert s["val_is_train"] is True
    assert sorted(s["train"]) == sorted(s["val"])          # mirrored
