#!/usr/bin/env python3
"""
test_stage2_metafeatures.py — prove the Stage-2 meta-feature fix without a GPU.

The fix in training.py:train_stage2_specialists() cannot be exercised end to end
without days of training, so the construction LOGIC is replicated here against
synthetic specialists whose behaviour we control exactly. That is enough to
establish the three properties that matter:

  1. the OLD sparse fill leaks the label, and the guard catches it;
  2. the NEW dense fill does not leak, and the guard passes it;
  3. under the new fill, a meta-learner's score tracks genuine specialist
     skill -- near chance when specialists are uninformative, high when they
     are informative -- instead of being pinned near 1.0 regardless.

Property 3 is the one that matters. A construction that scores ~0.99 whether or
not the specialists work is not measuring the specialists.

Run:  python3 test_stage2_metafeatures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_CLS, N_PER, SEED = 10, 120, 7


def _guard():
    """Load assert_meta_features_dense from training.py without importing
    torch / torch_geometric (unavailable off the training host)."""
    src = Path(__file__).resolve().parent.joinpath("training.py").read_text()
    start = src.index("def sanitize_meta")
    end = src.index("# ─────", start)
    ns: dict = {"np": np}
    exec(src[start:end], ns)
    return ns["assert_meta_features_dense"]


def synth(skill: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Dense matrix as the FIXED code builds it: specialist j scores every
    sample. `skill` in [0,1] interpolates from useless to near-perfect."""
    y = np.repeat(np.arange(N_CLS), N_PER)
    X = rng.uniform(0.0, 0.35, size=(len(y), N_CLS))
    hit = rng.uniform(0.55, 0.99, size=len(y))
    inform = rng.random(len(y)) < skill
    X[np.arange(len(y))[inform], y[inform]] = hit[inform]
    return X, y


def sparsify(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix as the BUGGY code built it: 0.5 everywhere except own class."""
    S = np.full_like(X, 0.5)
    S[np.arange(len(y)), y] = X[np.arange(len(y)), y]
    return S


def macro_f1(X, y, rng) -> float:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    import warnings
    warnings.filterwarnings("ignore")

    pred = np.zeros_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               verbose=-1, n_jobs=2)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(f1_score(y, pred, average="macro"))


def main() -> int:
    guard = _guard()
    rng = np.random.default_rng(SEED)
    failures = []

    # ── 1 & 2: the guard's verdict on each construction ──────────────────
    X, y = synth(0.85, rng)
    try:
        guard(sparsify(X, y), y, "sparse")
        failures.append("guard did NOT catch the sparse (buggy) construction")
    except AssertionError:
        print("[1] guard rejects the sparse construction            PASS")

    try:
        guard(X, y, "dense")
        print("[2] guard accepts the dense construction             PASS")
    except AssertionError as e:
        failures.append(f"guard wrongly rejected the dense matrix: {e}")

    # A resumed run skips specialists whose .pt already exists, so their columns
    # are never written and stay at the initialiser. Neither the culprit check
    # nor the shortcut check sees this -- an all-sentinel matrix has no culprit
    # rows and a chance-level shortcut -- so it needs its own test.
    try:
        guard(np.full_like(X, 0.5), y, "all-sentinel")
        failures.append("guard did NOT catch an all-sentinel (degenerate) matrix")
    except AssertionError:
        print("[3] guard rejects an all-sentinel matrix             PASS")

    half = X.copy()
    half[:, : N_CLS // 2 + 1] = 0.5
    try:
        guard(half, y, "half-constant")
        failures.append("guard did NOT catch a majority-constant matrix")
    except AssertionError:
        print("[4] guard rejects a majority-constant matrix         PASS")

    # ── 5: does the score track real specialist skill? ───────────────────
    print("\n[5] meta-learner macro F1 vs. true specialist skill")
    print(f"    {'skill':>6s} {'sparse (buggy)':>16s} {'dense (fixed)':>15s}")
    rows = []
    for skill in (0.0, 0.35, 0.85):
        Xs, ys = synth(skill, np.random.default_rng(SEED + int(skill * 100)))
        f_sparse = macro_f1(sparsify(Xs, ys), ys, rng)
        f_dense = macro_f1(Xs, ys, rng)
        rows.append((skill, f_sparse, f_dense))
        print(f"    {skill:6.2f} {f_sparse:16.4f} {f_dense:15.4f}")

    chance = 1.0 / N_CLS
    # The buggy construction should score high even when specialists are
    # useless -- that is precisely the defect.
    if rows[0][1] < 0.90:
        failures.append(f"sparse at skill=0 scored {rows[0][1]:.3f}; expected "
                        f">0.90 (the leak should mask total incompetence)")
    # The fixed construction must be near chance when specialists are useless.
    if rows[0][2] > chance + 0.10:
        failures.append(f"dense at skill=0 scored {rows[0][2]:.3f}; expected "
                        f"<={chance + 0.10:.2f} (near chance)")
    # ...and must rise with skill.
    if not rows[0][2] < rows[1][2] < rows[2][2]:
        failures.append("dense score does not increase monotonically with skill")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all checks passed — the dense construction measures the "
          "specialists;\nthe sparse one measures nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
