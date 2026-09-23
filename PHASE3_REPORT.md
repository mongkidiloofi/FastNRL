# Phase 3 report: Popper comparison and transfer

## Outcome

Phase 3's bounded baseline comparison and secondary transfer experiment are complete. The experiments are reproducible through [run_experiment.py](/home/tasin/Documents/For%20Thesis/Final%20Thesis/run_experiment.py), with one JSON result per seed, aggregate JSON, and aggregate CSV files.

The result is a tie on the deterministic `aaa_small` maze under this harness: Popper and the untouched Q-learning control both reached the goal in every evaluation run. This is not evidence that Popper is better than Q-learning or ILASP.

## Protocol

- Baseline maze: `aaa_small` (`vgdl_aaa_small-v0`).
- Seeds: 0--29 (`n=30`) for both Popper and Q-learning.
- Training/evaluation budget: 100 episodes, 250 steps per episode.
- Q-learning: the repository implementation and its protocol were left unchanged; the Phase 3 harness records a final greedy evaluation and learning-curve metrics separately.
- Popper: 10 bounded random-exploration episodes, one induction call, projection into the existing Clingo planner, then repeated evaluation using the requested 100-episode/250-step reporting budget.
- Popper's one-shot collection is a deliberate bounded harness. It is not a call-for-call reproduction of the original online ILASP loop; this limitation is recorded in every Popper result.

## Baseline results

`spread` is the population standard deviation recorded by the harness.

| Backend | Seeds | Success rate mean ± spread | Mean reward ± spread | Mean steps ± spread |
|---|---:|---:|---:|---:|
| Popper | 30 | 1.000 ± 0.000 | 4.000 ± 0.000 | 7.000 ± 0.000 |
| Q-learning | 30 | 1.000 ± 0.000 | 4.000 ± 0.000 | 7.000 ± 0.000 |

The Popper induction time was 3.997 ± 1.234 seconds per seed. Its planned action sequence was 7 steps in every run. The reward of 4 is consistent with six intermediate `-1` rewards followed by the terminal `+10` reward.

Machine-readable outputs:

- [Popper baseline summary](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/baseline/popper/summary.json)
- [Popper baseline CSV](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/baseline/popper/runs.csv)
- [Q-learning baseline summary](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/baseline/q_learning/summary.json)
- [Q-learning baseline CSV](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/baseline/q_learning/runs.csv)

## Transfer experiment

Popper was trained on `aaa_small` and evaluated on the existing registered `experiment3_after` maze for 5 seeds. All five runs succeeded with a 21-step plan: success rate 1.000 ± 0.000, reward -10.000 ± 0.000, and 21.000 ± 0.000 steps. Induction time was 3.474 ± 0.577 seconds.

The repository's `aaa_medium` registration could not be used because it refers to a missing `aaa_medium_lvl0.txt`; only a differently named copy is present. `experiment3_after` was therefore used as the available similar maze and is explicitly recorded as such. This is not claimed to reproduce the report's exact transfer pairing. A cross-maze Q-learning transfer comparison was not run; the harness marks Q-learning transfer as training-only rather than fabricating a transfer result.

- [Transfer summary](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/transfer/popper/summary.json)
- [Transfer CSV](/home/tasin/Documents/For%20Thesis/Final%20Thesis/phase3_results/transfer/popper/runs.csv)

## ILASP reference status

ILASP2i was not used to generate a new comparison sample. Phase 0 established that the binary launches, but the real context-dependent `symbolic-rl` tasks fail with `Poco::WriteFileException` in this environment. Consequently, the original report remains the published ILASP reference point; no new ILASP number is claimed here. The CLI's `--backend ilasp` path is intentionally reference-only and reports this status instead of inventing output.

Popper is pinned to v3.1.0, commit `227c30f`, with the compatibility fix documented in [PHASE2_REPORT.md](/home/tasin/Documents/For%20Thesis/Final%20Thesis/PHASE2_REPORT.md). The original ILASP source path and Q-learning source remain alongside the adapter; neither was overwritten.

## Is this the last phase, and is the code complete?

Phase 3 is the last numbered research phase in the requested plan. The remaining reporting/reproducibility work is part of the Phase 3 deliverable, not a separate Phase 4. Transfer is the optional secondary experiment within Phase 3.

The code is complete to test the Popper MVP and to reproduce the documented bounded baseline and transfer runs, after activating the Phase 0 SWI-Prolog/Popper runtime paths. For example:

```bash
python run_experiment.py --backend popper --experiment baseline --seed 0 --runs 30
python run_experiment.py --backend q_learning --experiment baseline --seed 0 --runs 30
python run_experiment.py --backend popper --experiment transfer --seed 0 --runs 5
```

It is not complete if “full” means an exact historical ILASP-vs-Popper reproduction: ILASP2i cannot currently rerun its real tasks here, Popper uses bounded one-shot induction rather than an exact online ILASP call schedule, and the transfer maze differs from the unavailable `aaa_medium` registration. Those are documented experimental limitations, not hidden behind a positive result. The defensible thesis claim at this point is that the Popper replacement works end-to-end on the repository maze and matches the Q-learning control on the tested deterministic baseline under the stated bounded protocol.
