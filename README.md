# grapheval

Measuring what learning-based vulnerability detection actually reports.

A code property graph turns a function into a graph of its syntax, control flow
and data dependencies; a graph neural network then classifies it. Accuracy in
this literature is high and rising. This work rebuilds that pipeline over
131,903 real vulnerabilities from 949 open-source projects and varies one factor
at a time to find out what the reported numbers are measuring.

Two choices that papers in this area do not report turn out to move results more
than the architectural differences between the systems being compared.

**The split protocol.** Holding corpus, architecture, features and seed fixed,
and restricting only what the test set may share with training, ROC-AUC falls
from 0.796 to 0.585. Measured against the margin above a coin flip that is 71%
of the usable signal: 47% attributable to duplicated functions, 24% to training
and testing on the same projects, 29% surviving every control.

**The training configuration.** Fixing the recipe by grid search on the
validation partition alone, with test withheld until the ranking was settled, is
worth a further 0.032 ROC-AUC — more than any representational choice measured
here. It also decides whether other choices register at all: five-fold averaging
gains 0.033 under library defaults and nothing under the selected recipe, and
the same holds for making graph edges traversable in both directions. Across the
twelve configurations searched, validation ranks test only weakly, so roughly a
point of ROC-AUC is unreachable without consulting the test partition.

Two further results do not depend on either choice. Relation labels — knowing an
edge carries data dependence rather than syntactic containment — are worth
0.002 (95% CI −0.005 to +0.010) against a baseline that holds edge direction
fixed. And on the 2,067 vulnerable/patched function pairs held in the test
partition, where project and coding style cancel by construction, no
configuration separates a function from its own fix at better than chance.

## Layout

| path | contents |
|---|---|
| `scripts/` | training, ablation and analysis code |
| `splits/` | split definitions for all three protocols, with disjointness audits |
| `results/` | every run artifact the reported numbers are computed from |
| `corpus/` | manifest: function id, label, CWE, repository, commit, clone family |

### Corpus versus splits

The manifest in `corpus/` describes 131,903 functions. The split manifests
contain **99,569**. The generator keeps a vulnerable record only if its CWE
family has at least 500 members, and drops Devign entirely because it carries no
CWE labels. Both filters exist for the ten-way CWE-family task; the binary
experiments reuse the same `_family` splits and inherit them. Per source:

| source | in corpus | in splits |
|---|---|---|
| BigVul | 89,273 | 86,143 |
| CVEfixes | 15,346 | 13,426 |
| Devign | 27,284 | 0 |

Prevalence is 23.4% across the corpus and 13.4% across the splits. Every
protocol uses the identical 99,569 functions, so comparisons between them are
controlled; the filtered population is what they describe.

Code property graphs are not redistributed. They are derived data, regenerated
from the public BigVul, CVEfixes and Devign corpora by `build_dataset_*.py`,
which needs Joern 4.0 and about a day of wall time.

## Reading the results

Every reported figure derives from `results/*.json`. The suffix says which
training configuration produced it, and that distinction is load-bearing: the
same comparison can change sign between them.

- `*_stage1.json` — five-fold ensemble, library defaults.
  `["ensemble_mean_only"]["roc_auc"]`; `["per_fold_roc_auc"]` holds the members.
- `*_single.json` — one model on the whole training partition.
  `["single_model"]["roc_auc"]`.
- `*_fullsingle.json` — one model on the whole partition under the *default*
  recipe; the controlled arm for the ensembling comparison.
- `*_ensnew.json` — five-fold ensemble under the *selected* recipe.
- `pair_slice_*.json` — paired vulnerable/patched evaluation. Files carrying a
  `_disposition` key use the corrected three-bucket slicing and store per-pair
  win indicators; earlier versions, kept under
  `results/pairslice_pre_disposition/`, used an `if`/`elif` with no `else` that
  silently dropped the 561 pairs whose members are near-identical in both node
  text and edges.

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

A citation will be added on publication.
