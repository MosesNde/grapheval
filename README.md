# GraphEval

**Evaluation Protocol and Graph Encoding in Vulnerability Detection with Code Property Graphs**

GraphEval is the replication repository for an empirical study of learning-based source-code vulnerability detection. The study evaluates how **data-partitioning protocols**, **code property graph (CPG) encoding**, and **training procedures** affect the performance of graph neural networks (GNNs). It also tests whether a graph-based detector can distinguish vulnerable functions from their patched counterparts.

The associated manuscript is being prepared for submission to *Empirical Software Engineering (EMSE)*. This repository is a **research artifact**, not a deployable vulnerability scanner.

## Research questions

- **RQ1 — Evaluation protocol:** How does performance change as training–test overlap is progressively restricted?
- **RQ2 — Graph encoding:** What do graph connectivity, edge direction, and edge-relation labels contribute within a controlled pipeline?
- **RQ3 — Learned representations:** What information can the network extract from graph structure, and does that translate into patch-sensitive discrimination?
- **RQ4 — External baseline:** How does the graph pipeline compare with LineVul on the same partitions?

A separate factorial analysis evaluates **training recipe × model aggregation**.

## Dataset and experimental scope

The assembled corpus contains **131,903 functions from 949 repositories**, drawn from BigVul, CVEfixes, and Devign. The main experiments use a **filtered subset of 99,569 functions from 701 repositories**, including **13,386 vulnerable functions**:

| Included source | Functions | Vulnerable |
| --- | ---: | ---: |
| BigVul | 86,143 | 7,633 |
| CVEfixes | 13,426 | 5,753 |
| **Experimental total** | **99,569** | **13,386** |

The filtering was inherited from corpus preparation involving CWE-family support: vulnerability families with fewer than 500 examples were excluded, and Devign was excluded because the required CWE-family metadata were unavailable. **The manuscript reports binary vulnerable/non-vulnerable detection, not ten-way CWE classification.** This filtering limits the population to which the binary results generalize.

The three principal partitioning protocols use the **same 99,569 functions**, but assign them differently:

| Protocol | Separation enforced between training and test |
| --- | --- |
| `random` | Stratified function-level split; pair, clone, commit, and repository overlap can remain |
| `group` | Vulnerable/patched pairs, commits, and clone families do not cross partitions; repositories can be shared |
| `repo` | Repositories are disjoint, which also prevents the audited pair, commit, and clone-family overlap |

The repository-disjoint protocol changes **which repositories and functions appear in each partition**, not just the amount of overlap. Differences between protocols are therefore **protocol-associated effects**, not estimates of the separate causal contribution of duplicates, patches, or repository sharing. One split assignment was evaluated per protocol; seed-level uncertainty does not measure variation over alternative split assignments.

## Selected findings

These are results for the configurations specified below, not guarantees about all graph-based vulnerability detectors.

### Protocol and encoding sensitivity

- Under the **inherited, directed-edge, five-fold GAT procedure**, ROC-AUC is **0.796** with `random` and **0.585** with `repo`. Approximately **71% of the random-split margin above chance** does not persist across that protocol transition. This reduction also reflects changes in the test-partition composition.
- With that inherited GAT procedure on `repo`, adding reverse edges changes ROC-AUC from **0.585 to 0.618** (**+0.033**). The benefit does **not** persist under every training configuration; under the alternative single-model procedure, the directed-to-bidirectional difference is **−0.006**.
- On a matched bidirectional comparison, distinguishing the four edge-relation types has a pooled effect of approximately **+0.002 ROC-AUC** (95% CI **[−0.005, +0.010]**) across the three edge-conditioned architectures.

### Training recipe and model aggregation

All four cells below are **GAT, bidirectional edges, repository-disjoint evaluation**, with five training seeds. A single model trains on the whole training partition; each five-fold ensemble member trains on four-fifths of it.

| Model procedure | Inherited recipe | Validation-selected recipe |
| --- | ---: | ---: |
| Single model | 0.585 ± 0.018 | 0.650 ± 0.004 |
| Five-fold ensemble | 0.618 ± 0.010 | 0.651 ± 0.003 |

The selected recipe **changes several settings together** (loss, sampling, effective batch size, schedule, and precision). Thus its +0.065 single-model gain is a **bundled training-configuration comparison**, not the isolated effect of one hyperparameter. Likewise, the single-versus-ensemble comparison changes both aggregation and the amount of data seen by each constituent model.

**Do not conflate the two different 0.585 values:** the protocol comparison's 0.585 is a **directed five-fold** GAT; the factorial table's 0.585 is a **bidirectional single-model** GAT. They happen to have the same rounded ROC-AUC.

### Structural controls and vulnerable/patched pairs

Structural positive controls demonstrate that the pipeline can use graph connectivity on auxiliary edge-defined tasks. These controls do **not** establish that it recognizes subtle vulnerability repairs or performs line-level localization.

The manuscript reports a **five-run tie-aware** vulnerable/patched analysis over **2,067 test-set pairs**. A correct ordering scores 1, an incorrect ordering 0, and an **exact score tie** 0.5:

| Pair category | Pairs | Graph | Separately trained edge-free model |
| --- | ---: | ---: | ---: |
| Structure-focused | 165 | 0.519 ± 0.063 | 0.402 ± 0.038 |
| Text-different | 1,341 | 0.468 ± 0.059 | 0.511 ± 0.073 |
| Near-identical signatures | 561 | 0.501 ± 0.021 | 0.472 ± 0.011 |

These categories use **multiset Jaccard similarities** of node text and edge signatures, with a threshold of 0.95. “Near-identical” means near-identical under these extracted signatures, not necessarily identical programs. The edge-free arm in this experiment is **trained separately**; it should not be confused with the representation-probe experiment, which removes edges from an already-trained model at inference.

The graph model's 0.519 accuracy on structure-focused pairs has a five-run 95% interval of **[0.440, 0.597]**, which includes 0.5. Its **relative gain** over the edge-free configuration (+0.116, Holm-adjusted p = 0.0153) therefore does **not** imply reliable absolute patch discrimination. The larger text-different slice shows a nonsignificant negative relative difference. Exact ties are particularly important in the near-identical slice.

### External comparison

LineVul achieves **0.792 ROC-AUC** on the repository-disjoint test partition. At a **1% false-positive rate**, the inherited five-fold GAT has recall **0.016**, the best measured single-model GIN has recall **0.026 ± 0.006**, and LineVul has recall **0.278**. This is **not a representation-only comparison**: LineVul fine-tunes CodeBERT end to end, whereas the principal graph configurations use frozen GraphCodeBERT node features.

## Repository layout

```text
grapheval/
├── README.md
├── corpus/
│   ├── cwe_corpus.jsonl.gz       # Assembled corpus manifest
│   └── cwe_corpus_report.json
├── splits/                       # Released partition manifests
├── results/                      # JSON run artifacts and analyses
└── scripts/                      # Corpus preparation, models, runs, and checks
```

The repository does **not** contain the original vulnerability datasets, generated CPG DOT files, model feature caches, or a complete locked software environment. Some scripts still use machine-specific paths under `/home/moses`; see **Reproduction requirements** below.

### Main code paths

| Purpose | Entry point |
| --- | --- |
| Assemble BigVul/CVEfixes-derived data | `scripts/build_dataset_bigvul.py`, `scripts/build_dataset_cvefixes.py`, `scripts/build_cwe_corpus.py` |
| Generate and audit partition manifests | `scripts/make_splits.py` |
| Shared graph parsing, features, and model code | `scripts/training.py` |
| Inherited five-fold GNN procedure | `scripts/train_cwe.py --stage 1` |
| Single-model training | `scripts/train_single.py` |
| Inherited single-model control | `scripts/train_fullsingle.py` |
| Selected-recipe five-fold ensemble | `scripts/train_ensemble_newrecipe.py` |
| Structural positive controls | `scripts/structural_control.py` |
| Vulnerable/patched evaluation | `scripts/pair_slice_test.py` |
| Aggregate tie-aware pair results | `scripts/analyze_pairslice_tieaware.py` |
| Independent pipeline checks | `scripts/verify_pipeline.py` |

Some auxiliary and historical scripts in `scripts/` reflect other ablations or earlier experiments and are **not** all part of the final manuscript. In particular, the manuscript does not report the older Stage 2 ten-way CWE experiment.

## Mapping results to artifacts

A filename alone is not a substitute for checking the recorded configuration. In particular, `*_single.json` files can refer to a single model under different recipe settings. Inspect each artifact's configuration fields before combining runs.

| Manuscript analysis | Relevant released artifacts |
| --- | --- |
| Protocol comparisons and architectures | `results/*_stage1.json`, corresponding `*_single.json` files; see seed suffixes and exclusion noted in the manuscript |
| Reverse edges | Matching directed and `-rev` run files under `results/` |
| Relation labels | Matched `repo-*-rev*_stage1.json` and `repo-*-euntyped-rev*_stage1.json` runs |
| Positive controls | `results/structural_control_cfg-cycle_repo-s*.json`, `results/structural_control_ddg-density_repo-s*.json` |
| Training factorial, inherited arm | `results/repo-gat-rev-s*_fullsingle.json` and the corresponding five-fold `-rev*_stage1.json` runs |
| Training factorial, selected arm | `results/repo-gat-rev-s*_ensnew-valsel.json` and selected single-model `-valsel_single.json` runs; **seed 1** is in the `-sw_e30_a8_lr2e-4_single.json` sweep output |
| Manuscript's five-run paired table | `results/pair_slice_repo-s{0,1,2,3,4}-tieaware.json` |
| Later full-precision paired reruns | `results/pair_slice_repo-s{0,1,2,3,4}-exact.json` |
| LineVul comparison | `results/linevul_official_stage1.json` |

`*_ensnew.json` **without** `-valsel` uses an earlier intermediate recipe (15 epochs by default); it is **not** the validation-selected ensemble reported in the factorial table.

### The two paired-result sets are not interchangeable

The five `-tieaware.json` runs contain **win/loss/tie counts calculated before their per-pair scores were rounded**. Those recorded counts support the five-run manuscript table and its seed-level tests. Recomputing individual outcomes from the rounded six-decimal scores can introduce **artificial ties**, particularly on near-identical pairs; do not use those rounded scores to claim an exact pair-level bootstrap.

The five later `-exact.json` files are **present in `results/`**. They store full-precision per-pair scores and explicit outcomes for reproducible pair-level reanalysis. They were generated by **additional runs**, not by restoring the lost precision of the earlier runs. Because training can be nondeterministic even at the same nominal seed, their results need not equal the original five-run manuscript values.

`analyze_pairslice_tieaware.py` defaults to `--tag exact` and also supports `--tag tieaware` and `--pool exact,tieaware`. **The manuscript's published five-run values should be checked using `--tag tieaware`; do not silently replace them with the `exact` results or a ten-run pool.** If the manuscript is ever revised to use the ten-run pool, the table, variability, confidence intervals, significance tests, and methodological description must all be updated together. Runs with reused seed labels should not automatically be treated as statistically independent merely because their outputs differ.

## Reproduction requirements

1. Obtain the original **BigVul and CVEfixes** distributions from their respective sources. The corpus manifest is metadata and does not replace source-code/CPG artifacts.
2. Install a compatible Python environment with the dependencies imported by the scripts, including **PyTorch, PyTorch Geometric, Transformers, NumPy, SciPy, scikit-learn**, and additional packages required by specific analyses. The repository currently has **no pinned `requirements.txt` or lockfile**; consult the run artifacts' environment metadata where available. GPU-enabled training was used for the main runs.
3. Install **Joern 4.0** if regenerating the CPGs. Generation and feature extraction can be resource- and time-intensive; the original CPG files and node-feature cache are not redistributed.
4. Update hard-coded paths such as `/home/moses/TSE_Journal`, `/home/moses/venv/bin/python`, and `/home/moses/graph_cache_cwe` in the relevant Python scripts and `run_*.sh` launchers. The LineVul launcher additionally assumes a separately installed copy of LineVul and exported baseline CSV files.
5. Use the **released split files** when reproducing the paper's specific partition assignment; rerunning split construction can produce a different assignment and is a distinct experiment.

For example, after cloning and obtaining the required external data, the model drivers expect the corpus and partition files under `data/`, not directly under `corpus/` and `splits/`:

```bash
mkdir -p data runs
# Decompress the supplied manifest:
gzip -dc corpus/cwe_corpus.jsonl.gz > data/cwe_corpus.jsonl
# Copy the exact split manifests used by the paper:
cp splits/splits_{random,group,repo}_family.json data/
```

**These preparation steps do not by themselves make training runnable.** Paths stored in the manifest must resolve to the regenerated CPGs, and the feature-cache and environment paths must also be configured. Do not alter split IDs or graph encodings unintentionally while setting up a reproduction.

### Check the manuscript's paired table without retraining

The following uses only the Python standard library and the five original count-based artifacts in `results/`; it does **not** reconstruct ties from rounded scores:

```bash
python - <<'PY'
from pathlib import Path
from statistics import mean, stdev
import json

paths = sorted(Path('results').glob('pair_slice_repo-s?-tieaware.json'))
assert len(paths) == 5, f'Expected five manuscript runs, found {len(paths)}'
runs = [json.loads(p.read_text()) for p in paths]

for key in ('structure-only', 'text-differs', 'near-identical'):
    n = runs[0][key]['n_pairs']
    def rates(arm):
        return [(r[key][f'{arm}_wins'] + 0.5*r[key][f'{arm}_ties']) / n
                for r in runs]
    g, ng = rates('graph'), rates('nograph')
    print(f'{key:16s} n={n:4d} | '
          f'graph={mean(g):.3f} ± {stdev(g):.3f} | '
          f'no graph={mean(ng):.3f} ± {stdev(ng):.3f} | '
          f'delta={mean(a-b for a,b in zip(g,ng)):+.3f}')
PY
```

To run the full statistical analysis, first change `ROOT` in `scripts/analyze_pairslice_tieaware.py` from its author-specific absolute path to your checkout directory, and copy the desired five files from `results/` into the `runs/` directory the script reads:

```bash
cp results/pair_slice_repo-s?-tieaware.json runs/
python scripts/analyze_pairslice_tieaware.py --tag tieaware

# Separate full-precision sensitivity/reanalysis (not the paper's original five runs):
cp results/pair_slice_repo-s?-exact.json runs/
python scripts/analyze_pairslice_tieaware.py --tag exact
```

The analysis needs NumPy and SciPy. Its run-level intervals measure **training-run variability**; its pair-level bootstrap measures **sampling variability over evaluated pairs**. Neither measures uncertainty over different repository-disjoint split assignments.

### Training entry points

Once the datasets, DOT paths, cached features, and environment are configured, representative commands are:

```bash
# Inherited GNN procedure on the repository-disjoint split
python scripts/train_cwe.py --stage 1 --mode repo --cache /path/to/graph_cache

# Selected recipe: single model, GAT, bidirectional edges
python scripts/train_single.py --mode repo --backbone gat --reverse-edges \
  --seed 0 --epochs 30 --batch-size 16 --accum 8 --lr 2e-4 \
  --tag valsel --cache /path/to/graph_cache
```

Other experiment launchers are in `scripts/run_*.sh`, but many retain absolute paths and GPU assumptions from the original research machine; **inspect and adapt them before execution**. The `run_*.sh` job queues rely on `scripts/single_jobs.txt` or `scripts/repopulate_jobs.txt`, as appropriate.

## Interpretation and limitations

- The headline **71%** is a **protocol-associated reduction**, not proof that overlap alone caused 71% of the performance change.
- The strongest reverse-edge gain is **configuration-dependent**, not a universal advantage of bidirectionality.
- Relation-label results indicate little measured benefit **under the specific matched comparisons**, not that relation types are never useful.
- Relative improvements on structure-focused pairs do not establish reliable absolute vulnerability/patch discrimination or statement-level localization.
- The primary experiments use **one split assignment per protocol**; reported standard deviations across training runs do not estimate split-selection uncertainty.
- The dataset and its CWE-derived filtering, preprocessing choices, graph encoding, and frozen-feature regime limit generalization to other languages, corpora, and detector implementations.

## Artifact availability and citation

The released code, corpus manifest, split definitions, and saved result artifacts are in this public repository. **Raw upstream datasets, generated CPGs, cached node embeddings, and a ready-to-run locked environment are not included.** Readers should obtain the upstream data from their original providers and respect their licenses and redistribution terms.

A formal citation and versioned replication-package release can be added once the manuscript has a publication record. When reporting or reproducing results, cite the repository **commit or release tag**, not an unversioned branch, so the scripts, result artifacts, and README refer to the same state.
