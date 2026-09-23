# Phase 2: Popper transition backend and single-maze loop

Status: minimum viable Popper backend complete and verified on the original `aaa_small` maze. Phase 3 comparison runs have not been started.

## Implementation

The new backend is [lib/popper_backend.py](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/lib/popper_backend.py). It exposes these boundaries:

```python
learn_transition_hypothesis(
    examples,
    background_knowledge,
    bias,
    *,
    popper_root=None,
    output_dir=None,
    timeout=120,
) -> str

convert_popper_hypothesis_to_planner(hypothesis) -> str
```

The first function returns a contextual Popper hypothesis. The second validates and projects it into the `state_at/2`, `action/2`, `adjacent/3`, and `wall/1` vocabulary expected by the original ASP planner. Unsupported syntax, link/teleport examples, mismatched direction bindings, missing literals, and unsafe projections fail explicitly.

The standalone verification runner is [phase2_smoke.py](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/phase2_smoke.py). It is separate from `main.py` and `q_learning.py`; neither original path was altered.

## Task translation

The adapter parses ordinary ILASP three-set examples:

```prolog
#pos({state_after(Target)},
     {state_after(Wrong), ...},
     {state_before(Source). action(Direction). wall(Wall), ...}).
```

Each example gets a private context identifier. For example, a right-action context is compiled into facts of the form:

```prolog
state_before(c0,cell(X,Y)).
step_right(c0,cell(X1,Y1),cell(X,Y)).
wall(c0,cell(WX,WY)).
open(c0,cell(OX,OY)).
pos(transition(c0,cell(TX,TY))).
neg(transition(c0,cell(NX,NY))).
```

`open/2` is a finite, explicit complement of the local wall facts over the cell range supplied in the ILASP task. This represents the original `not wall/1` condition without relying on a Popper negation feature that the pinned backend does not provide in its ordinary body bias.

The original ILASP bias has separate `adjacent(const(action),...)` and `action(const(action))` literals. Popper's standard bias does not directly encode the required equality between those two constants. The adapter therefore compiles their intended matched pair into `step_up/3`, `step_down/3`, `step_left/3`, and `step_right/3` background predicates. This removes nonsensical direction-mismatched clauses while preserving the intended four move/stay transition templates. The projection expands `step_right` back into `adjacent(right,...)` plus `action(right,...)`, and likewise for the other directions.

This constraint was necessary. An initial unconstrained translation learned a rule pairing `action_up` with `adjacent_right`; that rule fit the available examples but was not a safe transition model for the planner. The adapter now rejects such a mismatch by construction rather than silently accepting it.

## Verification protocol

The smoke runner:

1. Loads the original `vgdl_aaa_small-v0` py-vgdl environment.
2. Applies only runtime compatibility shims required by modern Gym, NumPy, and Pygame; the original environment source is unchanged.
3. Exhaustively replays reachable non-terminal paths and collects all four actions at each reached state.
4. Writes those observations as ILASP-style CDPIs with the original local-wall context and five candidate successor exclusions.
5. Compiles the task into Popper `bk.pl`, `exs.pl`, and `bias.pl` files.
6. Runs Popper `v3.1.0` at commit `227c30f` using the verified SWI-Prolog/Clingo toolchain.
7. Validates the returned eight-rule hypothesis and converts it to planner syntax.
8. Rebuilds the original Clingo planner at each execution step, adds only walls observed around the current position, obtains the first action at planner time `>= 1`, executes it in py-vgdl, and replans.

The pinned Popper checkout is `/home/tasin/Documents/For Thesis/Final Thesis/Popper`. Its Phase 0 compatibility edit to `popper/lp/test.pl` is retained; Phase 2 did not change that checkout.

## Result

The run was executed with output under `/tmp/symbolic-rl-phase2`.

| Quantity | Observed value |
|---|---:|
| Environment | `vgdl_aaa_small-v0` |
| Start | `(1,4)` |
| Goal | `(5,1)` |
| Reachable states collected | 7 |
| Transition examples | 28 |
| Terminal transitions | 1 |
| Popper true positives | 28 |
| Popper false negatives | 0 |
| Popper true negatives | 112 |
| Popper false positives | 0 |
| Learned transition rules | 8 |
| Executed planner actions | 7 |
| Goal reached | yes |

The executed action sequence was:

```text
right, right, right, right, up, up, up
```

The final environment position was `(5,1)`. The generated contextual hypothesis and the exact planner projection are saved in the temporary run directory as `popper_task/hypothesis.pl` and `planner_hypothesis.lp`; `result.json` contains the machine-readable summary.

This is a functional proof of the Phase 2 loop, not a performance claim. No induction time, mean, spread, ILASP comparison, or multi-seed result is claimed from this single smoke run.

## Reproduction command

The verified command was:

```text
PATH=/tmp/swi-root/usr/bin:$PATH \
LD_LIBRARY_PATH=/tmp/swi-root/usr/lib/x86_64-linux-gnu \
SWI_HOME_DIR=/tmp/swi-root/usr/lib/swi-prolog \
PYTHONPATH=/tmp/popper-pkgs:/home/tasin/Documents/For Thesis/Final Thesis/Popper \
python3 phase2_smoke.py --output-dir /tmp/symbolic-rl-phase2
```

The runtime dependencies used for the original maze were Gym `0.26.2`, Pygame `2.6.1`, and the repository's embedded py-vgdl. The legacy Gym registration, old four-return step API, Pygame key constants, and NumPy alias are handled inside the smoke runner only.

## Remaining limitations before Phase 3

- The backend currently supports the ordinary four-action transition task only. Link/teleport modes are deliberately rejected until they receive a separate bias and projection test.
- The `step_<direction>` compilation is a controlled bias restriction and must be documented in any later Popper/ILASP comparison. It is intended to preserve the original transition semantics, not to claim byte-for-byte equivalence of the two search spaces.
- The smoke runner uses exhaustive experience collection to establish the first end-to-end loop. It does not yet reproduce the original incremental episode schedule, epsilon exploration, or ILASP-call replacement schedule.
- `run_experiment.py`, JSON/CSV per-run evaluation, transfer learning, and the 30-run comparison belong to Phase 3 and later.
- The original ILASP rerun remains unavailable under the Phase 0 result; the published report remains the ILASP reference unless that status changes.
