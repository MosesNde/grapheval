# grapheval

Measuring what learning-based vulnerability detection actually reports.

A code property graph turns a function into a graph of its syntax, control flow
and data dependencies; a graph neural network then classifies it. Accuracy in
this literature is high and rising. This work rebuilds that pipeline over 99,569
functions from 701 open-source projects and varies one factor at a time to find
out what the reported numbers are measuring.

Two choices that papers in this area do not report turn out to move results more
than the architectural differences between the systems being compared.

**The split protocol.** Holding corpus, architecture, features and seed fixed,
and restricting only what the test set may share with training, ROC-AUC falls
from 0.796 to 0.585. Measured against the margin above a coin flip that is 71%
of the usable signal: 47% attributable to duplicated functions, 24% to training
and testing on the same projects, 29% surviving every control.

**The training configuration.** Fixing the recipe by grid search on the
validation partition alone, with test withheld until the ranking was settled, is
worth a further 0.065 ROC-AUC for a single model and 0.033 for the five-fold
ensemble, in both cases more than any representational choice measured here. It
also decides whether other choices register at all: five-fold averaging gains
0.033 under library defaults and nothing under the selected recipe, and the same
holds for making graph edges traversable in both directions. Across the twelve
configurations searched, validation ranks test only weakly, so roughly a point of
ROC-AUC is unreachable without consulting the test partition.

Two further results do not depend on either choice. Distinguishing the relation
labels, so that an edge carrying data dependence is told apart from one carrying
syntactic containment, is worth 0.002 (95% CI −0.005 to +0.010) against a
baseline that holds edge direction fixed. And on the 2,067 vulnerable/patched
function pairs held in the test partition, where project and coding style cancel
by construction, no
configuration separates a function from its own fix at better than chance.

## Layout

| path | contents |
|---|---|
| `scripts/` | training, ablation and analysis code |
| `splits/` | split definitions for all three protocols, with disjointness audits |
| `results/` | every run artifact the reported numbers are computed from |
| `corpus/` | manifest: function id, label, CWE, repository, commit, clone family |

### What the experiments actually use

Every result derives from **99,569 functions** across 701 projects, 13.4% of
them vulnerable:

| source | functions | vulnerable |
|---|---|---|
| BigVul | 86,143 | 7,633 |
| CVEfixes | 13,426 | 5,753 |

The manifest in `corpus/` is larger, at 131,903, because it records the corpus
as assembled before filtering. Two filters apply. A vulnerable record is kept
only if its CWE family has at least 500 members, which the ten-way CWE
classification requires and which the binary task inherits, so rare
vulnerability types are absent from both. The same restriction removes the
Devign subset entirely, since it carries no CWE labels; Devign appears in the
manifest but contributes to no result. Every protocol uses the identical 99,569
functions, so comparisons between them are controlled.

Code property graphs are not redistributed. They are derived data, regenerated
from the public BigVul and CVEfixes corpora by `build_dataset_*.py`,
which needs Joern 4.0 and about a day of wall time.

## Reading the results

Every reported figure derives from `results/*.json`. The suffix says which
training configuration produced it, and that distinction is load-bearing: the
same comparison can change sign between them.

- `*_stage1.json`: five-fold ensemble, library defaults.
  `["ensemble_mean_only"]["roc_auc"]`; `["per_fold_roc_auc"]` holds the members.
- `*_single.json`: one model on the whole training partition.
  `["single_model"]["roc_auc"]`.
- `*_fullsingle.json`: one model on the whole partition under the *default*
  recipe; the controlled arm for the ensembling comparison.
- `*_ensnew.json`: five-fold ensemble under `train_single.py`'s *defaults*
  (15 epochs, lr 5e-4), not the selected recipe.
- `*_ensnew-valsel.json`: five-fold ensemble under the *selected* recipe
  (30 epochs, effective batch 128, lr 2e-4). This is the matched counterpart to
  the `-valsel` single-model runs; the pair of suffixes exists because an earlier
  launcher passed no hyperparameters, so the two arms of the comparison ran under
  different recipes. Every file written since carries a `config` and `env` block
  recording the recipe, aggregation, checkpoint criterion and library versions.
- `pair_slice_*-tieaware.json`: paired vulnerable/patched evaluation, five
  seeds, with the disposition of every pair as win / loss / tie. Each file
  carries the strict rate (wins only, ties charged as failures), the tie-aware
  rate (a tie credited as half a win), a pair-level bootstrap interval, tie
  counts under both an exact and a 1e-6 definition, and the raw per-pair scores,
  so any tie rule or interval can be recomputed without retraining.
  `analyze_pairslice_tieaware.py` aggregates the runs, defaulting to the
  full-precision `-exact` files; pass `--tag tieaware` for the older set, or
  `--pool exact,tieaware` for the ten-run analysis the paper reports. Pooling is
  legitimate because a seed does not reproduce: `set_seed` does not make the
  CUDA scatter reductions in message passing deterministic, and the weighted
  sampler is built without an explicit generator, so the two sets of five runs
  at identical seeds and configuration are independent replications. They differ
  by up to 0.13 on a single cell. Point estimates and run-level tests use all
  ten runs; the pair-level bootstrap uses only the five that carry exact
  per-pair outcomes.
  It asserts that pairs are aligned across seeds by sample id rather than
  assuming it, and that reconstructed per-pair outcomes reproduce the counts the
  run booked at full precision. On rounded artifacts that assertion is relaxed
  and the per-cell displacement is printed instead. Ties
  matter on one slice only: where the patch leaves no trace in the graph, 22.6%
  of pairs receive the same score for both members, and charging those as
  failures is what had put that slice below chance. How a tie is defined is a
  reportable choice there: admitting differences within 1e-6 raises that slice's
  tie fraction to 64.1%, against 8.0% and 4.0% on the other two.

  These files store per-pair scores rounded to six decimals. Their win/loss/tie
  counts were computed before rounding and are exact, so point estimates and
  seed-level intervals are exact, but the pair-level bootstrap is rebuilt from
  the scores and is approximate: on the near-identical slice the scores are
  ~1e-3, so six decimals can read a difference of 1e-7 as a tie. The per-cell
  displacement of a recomputed rate runs from -0.002 to +0.030, largest on the
  edge-free structure-only cell, where 3 to 5 real ties per seed become 25 to
  47. On the near-identical graph cell 259 of 561 pairs cannot be resolved from
  the stored data at all. `pair_slice_repo-s*-exact.json`, produced by
  `run_pairslice_exact.sh`, supersedes them: full-precision scores, an explicit
  per-pair outcome code, and the whole test partition's scores with sample ids,
  so any slice definition or tie rule can be re-analysed without retraining.
- `pair_slice_*.json` without that suffix: the earlier runs, which stored only
  per-pair win booleans. Ties had already been folded into losses at write time
  and cannot be recovered from them. Superseded two-bucket versions are kept
  under `results/pairslice_pre_disposition/`: they used an `if`/`elif` with no
  `else` that silently dropped the 561 pairs whose members are near-identical in
  both node text and edges, and their per-seed values did not reproduce on
  repetition.

Tags read `{protocol}-{backbone}[-e{edges}][-rev][-s{seed}][-{tag}]`, where
protocol is `random`, `group` or `repo`, `-rev` makes every edge traversable in
reverse, and `-euntyped` collapses the four relation types into one.

The seed is the unit of replication; comparisons are paired t-tests over five
seeds. The five folds inside a run share a training partition and a seed, so
pooling them as independent observations inflates the sample fivefold.
`group-gat-s2` is excluded from the five-fold `group` cell as a reduced-batch
rerun after a CUDA failure.

## Rerunning

`train_cwe.py` is the five-fold arm, `train_single.py` the single-model arm, and
`train_fullsingle.py` with `train_ensemble_newrecipe.py` the controlled
procedure comparison. Use `splits/` directly for an exact reproduction, or
`make_splits.py` to regenerate. Scripts carry absolute paths under `/home/moses`;
adjust `ROOT` and `--cache` for your environment.

The `run_*.sh` launchers reproduce whole tables rather than single runs. Those
that sweep a queue read it from a job list, one `mode|extra args|tag` record per
line: `run_single_sweep.sh` reads `single_jobs.txt`, and the
`run_repopulate*.sh` family reads `repopulate_jobs.txt`.

A citation will be added on publication.
