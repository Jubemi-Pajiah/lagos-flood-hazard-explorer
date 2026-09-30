"""Land cover map from Sentinel-2 with a Random Forest trained on ESA WorldCover labels.

Features per pixel: the four 10 m bands (B02, B03, B04, B08) and four indices. No SWIR
band is used, so "MNDWI" and "NDBI" are both built from green and red and are exact
negatives of each other; they are kept as two features to match the original model.

Training pixels are sampled per class (``samples_per_class``), split into training and
holdout sets (``train_fraction``), and the holdout gives the accuracy figures. No
post-classification smoothing is applied.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from flood import preprocess
from flood.config import Region
from flood.io import Grid, align_classes, read_classes, write_json, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)

FEATURE_NAMES = ["B02", "B03", "B04", "B08", "NDVI", "NDWI", "MNDWI", "NDBI"]
_EPS = 1e-6
_CHUNK_ROWS = 500


def landcover_path(region: Region) -> Path:
    return region.out("landcover", "landcover.tif")


def probability_path(region: Region) -> Path:
    return region.out("landcover", "landcover_probability.tif")


def accuracy_path(region: Region) -> Path:
    return region.out("landcover", "accuracy.json")


def features(b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """Stack bands and indices into an (..., 8) float32 feature array."""
    b02, b03, b04, b08 = (b.astype("float32") for b in (b02, b03, b04, b08))
    ndvi = (b08 - b04) / (b08 + b04 + _EPS)
    ndwi = (b03 - b08) / (b03 + b08 + _EPS)
    mndwi = (b03 - b04) / (b03 + b04 + _EPS)
    ndbi = (b04 - b03) / (b04 + b03 + _EPS)
    return np.stack([b02, b03, b04, b08, ndvi, ndwi, mndwi, ndbi], axis=-1)


def label_array(worldcover: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Map WorldCover codes to model classes; unmapped codes become 0."""
    labels = np.zeros(worldcover.shape, dtype="uint8")
    for code, cls in mapping.items():
        labels[worldcover == code] = cls
    return labels


def stratified_split(
    labels: np.ndarray,
    usable: np.ndarray,
    classes: list[int],
    per_class: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to ``per_class`` flat indices per class and split them train/holdout.

    Uses a legacy ``RandomState`` so the draws match ``np.random.seed(seed)`` followed by
    ``np.random.choice`` calls in class order.
    """
    rng = np.random.RandomState(seed)
    train, holdout = [], []
    for cls in classes:
        pool = np.where((labels == cls) & usable)[0]
        if len(pool) == 0:
            log.warning("class %s has no usable pixels", cls)
            continue
        n = min(per_class, len(pool))
        chosen = rng.choice(pool, size=n, replace=False)
        split = int(train_fraction * n)
        train.extend(chosen[:split])
        holdout.extend(chosen[split:])
    return np.array(train), np.array(holdout)


def accuracy_summary(
    y_true: np.ndarray, y_pred: np.ndarray, classes: dict[int, str], model: RandomForestClassifier
) -> dict[str, Any]:
    """Overall accuracy, kappa, confusion matrix and per-class scores for the holdout."""
    ids = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    report = classification_report(
        y_true,
        y_pred,
        labels=ids,
        output_dict=True,
        target_names=list(classes.values()),
        zero_division=0,
    )
    return {
        "overall_accuracy": float(np.trace(cm) / cm.sum()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classes": list(classes.values()),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {k: float(v) for k, v in report[name].items()} for name in classes.values()
        },
        "holdout_pixels": int(len(y_true)),
        "feature_importance": dict(
            zip(FEATURE_NAMES, (float(v) for v in model.feature_importances_), strict=True)
        ),
    }


def run(region: Region) -> None:
    cfg = region["landcover"]
    classes = {int(k): v for k, v in cfg["classes"].items()}
    bands = {
        b: read_classes(preprocess.sentinel2_path(region, b))[0]
        for b in ("B02", "B03", "B04", "B08")
    }
    grid = Grid.of(preprocess.sentinel2_path(region, "B02"))
    worldcover = align_classes(preprocess.worldcover_path(region), grid)
    rows, cols = grid.shape

    valid = ((bands["B02"] > 0) & (bands["B08"] > 0)).ravel()
    labels = label_array(worldcover, {int(k): v for k, v in cfg["training_labels"].items()})
    labels = labels.ravel()
    del worldcover
    log.info("valid pixels: %s", f"{int(valid.sum()):,}")

    train_idx, test_idx = stratified_split(
        labels,
        valid & (labels > 0),
        list(classes),
        cfg["samples_per_class"],
        cfg["train_fraction"],
        cfg["random_seed"],
    )
    flat = {b: a.ravel() for b, a in bands.items()}

    def sample(idx: np.ndarray) -> np.ndarray:
        return features(flat["B02"][idx], flat["B03"][idx], flat["B04"][idx], flat["B08"][idx])

    rf_cfg = cfg["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=rf_cfg["class_weight"],
        random_state=cfg["random_seed"],
        n_jobs=-1,
    )
    model.fit(sample(train_idx), labels[train_idx])
    y_pred = model.predict(sample(test_idx))
    summary = accuracy_summary(labels[test_idx], y_pred, classes, model)
    summary["training_pixels"] = int(len(train_idx))
    log.info(
        "holdout overall accuracy %.3f, kappa %.3f", summary["overall_accuracy"], summary["kappa"]
    )

    lulc = np.zeros(rows * cols, dtype="uint8")
    keep_probability = bool(cfg.get("save_probability", False))
    prob = np.zeros(rows * cols, dtype="float32") if keep_probability else None
    for r0 in range(0, rows, _CHUNK_ROWS):
        i0, i1 = r0 * cols, min(r0 + _CHUNK_ROWS, rows) * cols
        v = valid[i0:i1]
        if not v.any():
            continue
        idx = np.arange(i0, i1)[v]
        feats = sample(idx)
        lulc[idx] = model.predict(feats)
        if prob is not None:
            prob[idx] = model.predict_proba(feats).max(axis=1)

    missing = sorted(set(classes) - set(np.unique(lulc[lulc > 0]).tolist()))
    if missing:
        raise RuntimeError(f"classes {missing} never predicted; check the training labels")

    write_raster(landcover_path(region), lulc.reshape(rows, cols), grid, "uint8")
    if prob is not None:
        write_raster(
            probability_path(region), np.where(valid, prob, np.nan).reshape(rows, cols), grid
        )

    write_json(accuracy_path(region), summary)


STAGE = Stage(
    name="landcover",
    run=run,
    inputs=lambda r: [
        preprocess.worldcover_path(r),
        *(preprocess.sentinel2_path(r, b) for b in ("B02", "B03", "B04", "B08")),
    ],
    outputs=lambda r: [landcover_path(r), accuracy_path(r)],
    config=("landcover_grid", "landcover"),
    description="Random Forest land cover from Sentinel-2 with WorldCover training labels",
)
