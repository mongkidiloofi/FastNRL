# Phase 1: symbolic-rl architecture and ILASP-to-Popper mapping

Status: complete for the source-reading and dataflow-tracing phase. No Popper adapter code has been written in Phase 1.

## Scope and source state

This document describes the original `921kiyo/symbolic-rl` implementation as checked out locally at commit `27168c0` on `master`. The source snapshot is a legacy Python 3 / OpenAI Gym / py-vgdl project. The Popper checkout is separate and is not modified by this Phase 1 work.

The report PDF is treated as the authoritative description of the original experimental protocol. The implementation is treated as authoritative for the actual current control flow and command lines. Where those differ, both are recorded below.

## Files and components read

- `README.md`: dependency and execution notes, including the ILASP2i requirement.
- `report.pdf`: experimental design, task semantics, ILASP bias, CDPI examples, planner, and reported results.
- `config.py`: filenames, time horizon, adjacency rules, and experiment constants.
- `main.py`: no-transfer learning loop and test-time planner execution.
- `q_learning.py`: unchanged Q-learning comparison baseline.
- `tl.py`: transfer-learning loop with an initial transition hypothesis.
- `noTL_goal.py`: no-transfer loop with a known goal.
- `lib/induction.py`: wall discovery, CDPI generation, ILASP invocation, and coverage checks.
- `lib/py_asp.py`: predicate constructors, ILASP-to-planner string conversion, and action encoding.
- `lib/abduction.py`: ASP planner construction, marker replacement, Clingo invocation, and answer-set parsing.
- `lib/helper.py`: environment/action helpers, wall checks, and experiment logging.
- `las_base.las`, `tl_pos.las`, and report examples: concrete bias and task-file formats.

Important source locations are also linked at the end of this document.

## End-to-end architecture

The main no-transfer path is:

```text
Gym/py-vgdl maze
  -> reset/step and coordinate observation
  -> random exploration or ASP-selected action
  -> observed transition (state_before, action, state_after, nearby walls)
  -> induction.generate_pos(...)
  -> output.las containing ILASP CDPIs
  -> induction.run_ILASP(...)
  -> ILASP hypothesis in state_before/state_after/action/adjacent/wall vocabulary
  -> py_asp.convert_las_asp(...)
  -> clingo.lp planner program
  -> clingo5 JSON answer set
  -> sorted action sequence
  -> environment step
```

The environment and Q-learning baseline are separate from the induction backend. The intended Phase 2 change is therefore at the `run_ILASP` / hypothesis interface boundary, with the rest of the planner and evaluation protocol preserved.

## Environment semantics

The report describes a deterministic grid maze. Coordinates identify cells; the available actions are `up`, `down`, `left`, and `right`. A non-terminal transition receives a negative reward and reaching the goal receives a positive reward. An episode terminates at the goal or at the time limit.

The source encodes the action integers consistently in `lib/helper.py` and `lib/py_asp.py`:

| Integer | Symbol | Meaning |
|---:|---|---|
| 0 | `up` | move toward decreasing y |
| 1 | `down` | move toward increasing y |
| 2 | `left` | move toward decreasing x |
| 3 | `right` | move toward increasing x |
| 4 | `non` | planner fallback/no-op marker; not part of the four-action policy |

The implementation uses `+10` for a terminal transition and `-1` otherwise when recording learning/test rewards. `config.py` sets the planning/evaluation horizon to `TIME_RANGE = 250`.

The source obtains all wall coordinates from the py-vgdl game in `induction.get_all_walls`. Only walls adjacent to the current state are added incrementally to the planner. This is distinct from the complete wall list used when checking which local walls surround a transition.

## Main learning path (`main.py`)

`main.k_learning` performs the no-transfer experiment.

1. It reads the environment dimensions and creates a `cell((x,y)).` range for the maze.
2. It removes/recreates `ground.lp`, `output.las`, `clingo.lp`, and the cache file, then calls `induction.copy_las_base` to put the cell facts, optional link modes, adjacency rules, and ILASP bias into `output.las`.
3. It resets the environment and records the current observer coordinate as `previous_state`.
4. Until a goal state has first been discovered, it takes random actions. Each observed transition is passed to `induction.generate_pos`.
5. A newly generated CDPI is appended to `output.las`. If the current hypothesis fails the coverage check, or no hypothesis exists, the code invokes ILASP and replaces the current hypothesis.
6. Once a goal has been found, the learned hypothesis is converted to planner syntax and inserted into `clingo.lp`. The planner receives the current start state, the goal, the known walls, and a 250-step time range.
7. On every planning step, Clingo returns an answer set. The code extracts the selected action, optionally overrides it with epsilon exploration, steps the environment, and generates another CDPI from the resulting transition.
8. If the new example is not covered, ILASP is called again. The new hypothesis is reconverted and replaces the hypothesis section in `clingo.lp`.
9. After each training episode, `run_experiment` evaluates the current planner policy. It runs Clingo from the observed test position, executes the returned plan, and records test statistics.

The two ILASP call sites in this loop are in the initial learning branch and in the subsequent planned/exploratory branch. They both use `induction.run_ILASP`; there is no second independent induction implementation hidden in the main loop.

## CDPI generation and the actual ILASP input

`induction.generate_pos` constructs a context-dependent positive example for each observed transition.

For a normal move, the task file receives a form equivalent to:

```prolog
#pos({state_after((next_x,next_y))},
     {state_after((wrong_x,wrong_y)), ...},
     {state_before((prev_x,prev_y)).
      action(direction).
      wall((adjacent_wall_x,adjacent_wall_y)). ...}).
```

The three sets have different roles:

- The inclusion set (`Einc`) contains the actual observed successor, unless the current hypothesis already predicts it.
- The exclusion set (`Eexc`) contains successor states that must not be predicted in this context. It includes observed alternatives and predictions made by the current hypothesis that disagree with the actual transition.
- The context contains the current state, the selected action, and the locally observed adjacent walls. The background adjacency rules and cell facts are supplied globally by the task file.

The report gives examples such as:

```prolog
#pos({state_after((2,3))},{},
     {state_before((1,3)). action(right).
      wall((0,3)). wall((1,4)).}).

#pos({state_after((3,3))},{},
     {state_before((3,3)). action(right).
      wall((3,2)). wall((4,3)). wall((3,4)).}).
```

The checked-in report example also shows an explicit exclusion set:

```prolog
#pos({state_after((4,4))},
     {state_after((4,3)),state_after((3,4)),
      state_after((5,4)),state_after((4,5))},
     {state_before((4,4)). action(right).
      wall((5,4)). wall((4,5)).}).
```

The source additionally handles a detected non-adjacent transition as a link/teleport case. It can emit two examples: one for the expected adjacent result and one for the observed destination, and it may add `link_start`/`link_dest` facts and modes. This is part of the transfer/link extension and should not be silently included in the simplest Phase 2 maze unless the original task requires it.

The source writes these examples to `output.las`, whose name comes from `config.LASFILE`. Several file operations use relative paths, so the original scripts effectively expect the current working directory to be the `symbolic-rl` repository directory.

## The original ILASP bias

`las_base.las` contains the following adjacency background rules and bias:

```prolog
adjacent(right,(X+1,Y),(X,Y)) :- cell((X,Y)),cell((X+1,Y)).
adjacent(left,(X,Y),(X+1,Y)) :- cell((X,Y)),cell((X+1,Y)).
adjacent(down,(X,Y+1),(X,Y)) :- cell((X,Y)),cell((X,Y+1)).
adjacent(up,(X,Y),(X,Y+1)) :- cell((X,Y)),cell((X,Y+1)).

#modeh(state_after(var(cell))).
#modeb(1, adjacent(const(action), var(cell), var(cell)), (positive)).
#modeb(1, state_before(var(cell)), (positive)).
#modeb(1, action(const(action)), (positive)).
#modeb(1, wall(var(cell))).
#max_penalty(50).
#constant(action,right/left/down/up).
```

The report explains that the intended hypothesis is an eight-rule transition model: for each direction, the agent moves to the adjacent cell when it is open and stays in place when the target cell is a wall. A representative rule is:

```prolog
state_after(V0) :- adjacent(right,V0,V1), state_before(V1),
                    action(right), not wall(V0).
```

The negative wall literal is important. The bias allows negation for the `wall` predicate, and the maximum penalty is set because the target hypothesis is larger than the default penalty bound.

The initial report example also shows that ILASP can learn a partial rule from one positive example, such as:

```prolog
state_after(V0) :- adjacent(right,V0,V1).
```

The loop then adds more context-dependent examples and refines the hypothesis.

## ILASP invocation and reference status

The implementation invokes ILASP through `subprocess.check_output` in `induction.run_ILASP`:

```text
ILASP --version=2i FILENAME.las -ml=10 -q -nc --clingo5
      --clingo "clingo5 --opt-strat=usc,stratify"
      [cache path] --max-rule-length=8
```

The report's documented command uses `-ml=8`, `--cached-ref=PATH`, and does not include the source's quiet flag in the same way. The checked-in code uses `-ml=10` and the option spelling `--cached-rel` in the constructed command. This discrepancy must be resolved or explicitly preserved when making a reproducibility claim; it is not safe to infer that the report command and current source command are equivalent.

Phase 0 established that the old ILASP2i executable could be launched with compatibility libraries, but the real context-dependent symbolic-rl task files failed in the current environment with a Poco write-file exception. Therefore this project should use the published report numbers as the ILASP reference point unless a later independent ILASP repair succeeds. The original ILASP path remains in the repository and is not being overwritten.

## Hypothesis consumption by the ASP planner

The planner is assembled in `lib/abduction.py` and stored in `clingo.lp`.

The base program contains:

```prolog
1 { action(down,T); action(up,T); action(right,T); action(left,T) } 1
    :- time(T), not finished(T).
time(0..TIME_RANGE).
cell(...).
#show state_at/2.
#show action/2.
#minimize { 1,X,T : action(X,T) }.
```

It also includes the four adjacency rules from `config.ADJACENT`. As the episode proceeds, the source adds facts such as `wall((x,y)).` for walls that have been observed near the current state. Marker blocks in `clingo.lp` are replaced for the start state, hypothesis, and goal:

- The start block contains `state_at((x,y),1).`.
- The hypothesis block contains the converted transition rules.
- The goal block contains the goal, finished/goal predicates, and a constraint preventing plans that do not reach the goal.

Clingo is invoked as:

```text
clingo5 --opt-strat=usc,stratify -n 0 clingo.lp --opt-mode=opt --outf=2
```

`abduction.run_clingo` parses the JSON answer sets and returns the last witness. `sort_planning` separates `state_at/2` and `action/2` atoms and orders them by time. `py_asp.extract_action` maps the planner action symbol back to the environment integer.

## Exact ILASP-to-planner conversion

`py_asp.convert_las_asp` currently converts the learned hypothesis by string replacement rather than parsing rules. Its intended mapping is:

| ILASP transition vocabulary | Planner vocabulary |
|---|---|
| `state_before(V)` | `state_at(V,T)` |
| `state_after(V)` | `state_at(V,T+1)` |
| an otherwise empty body | `time(T)` |
| `action(A)` | `action(A,T)` |
| `:-` | `:- time(T),` |

For example:

```prolog
state_after(V0) :- adjacent(right,V0,V1), state_before(V1),
                    action(right), not wall(V0).
```

becomes:

```prolog
state_at(V0,T+1) :- time(T), adjacent(right,V0,V1),
                     state_at(V1,T), action(right,T), not wall(V0).
```

This conversion assumes the exact variable names and textual shapes emitted by ILASP. It is not a general ASP parser. A Popper adapter must either emit a deliberately compatible hypothesis representation or replace this conversion with a checked, syntax-aware compiler. In either case, a real learned hypothesis must be loaded into `clingo.lp` and shown to produce a valid plan before claiming compatibility.

## Q-learning baseline and evaluation protocol

`q_learning.py` is an independent baseline and must remain unchanged. It uses four actions, epsilon-greedy exploration with epsilon `0.1`, learning rate `alpha = 0.5`, discount factor `1`, the same `-1/+10` reward convention, and a 250-step episode horizon. It performs a greedy test rollout after each training episode.

The report's main comparison protocol is 30 independent runs, 100 training episodes per run, and 250 steps per episode. The primary evaluation is the time/episodes needed to reach the optimal deterministic policy and the resulting success/reward behavior. The report also records cumulative induction time, induction-call counts, and average ILASP call time. A no-hypothesis/no-goal ILP policy is treated as receiving the time-limit penalty, rather than being silently omitted.

The reported reference observations include ILP reaching the optimal policy before roughly episode 40 and Q-learning around roughly episode 60 in the original experiment, plus reported ILASP timing/call statistics. These are published report values, not a new rerun in this environment.

## Transfer and known-goal paths

`tl.py` starts with a supplied transition hypothesis `h`, a supplied goal, and a planner that can act immediately. It then follows the same pattern: execute a planned or exploratory action, generate context-dependent examples, invoke ILASP when coverage fails, convert the hypothesis, and update the planner.

`noTL_goal.py` starts with a known goal but no transition hypothesis. It can therefore plan only after enough examples have induced a usable model.

In the current source snapshot, `tl_pos.las` is effectively empty. The active transfer seed is the hardcoded hypothesis at the bottom of `tl.py`, not a non-empty set of checked-in transfer examples. This should be reported explicitly if the transfer experiment is implemented later.

The report's later experiments add `link_start` and `link_dest` concepts for teleport/link behavior. They enlarge the bias and search space substantially and should remain a separate extension from the minimal ordinary-maze adapter.

## Popper adapter boundary identified by Phase 1

The clean replacement boundary is conceptually:

```text
ILASP-style examples + original bias
  -> context-preserving Popper task compiler
  -> Popper hypothesis
  -> validated compiler to planner-compatible ASP
```

The rest of the environment, experience collection, reward handling, planner, and evaluation protocol should depend only on the returned transition hypothesis, not on Popper internals.

The translation cannot be a mechanical rename from `#pos` to `pos`. There are four research-integrity hazards:

1. **Context isolation.** ILASP's `#pos(Einc,Eexc,Context)` evaluates each example in its own context. Popper's normal `pos(...)` and `neg(...)` examples use shared background knowledge. Putting all ILASP context facts into one global `bk.pl` would allow facts from one transition to explain another and would silently change the learning problem.

   The implemented preservation mechanism reifies a context identifier and compiles direction-paired action/adjacency facts, for example `state_before(C,Cell)`, `step_right(C,To,From)`, `wall(C,Cell)`, `open(C,Cell)`, and `transition(C,Cell)`. Each ILASP inclusion becomes a positive Popper example for its context, and each ILASP exclusion becomes a negative example for the same context. The compiler keeps context identifiers private to each example.

2. **ILASP constants versus Popper predicates.** `const(action)` in the ILASP mode declaration restricts the action argument to the four named directions. Popper's ordinary bias declarations do not directly provide the same ILASP constant-mode semantics. A naive variable action argument would give Popper a larger hypothesis space and make the comparison unfair.

   The implemented design encodes the four matched direction pairs as `step_up`, `step_down`, `step_left`, and `step_right`. This excludes direction-mismatched action/adjacency clauses that the raw Popper predicate vocabulary could otherwise express, while the planner projection expands each step relation back into the original action and adjacency atoms.

3. **Negative wall literals.** The original target uses `not wall(V)` and the ILASP bias explicitly permits the relevant negation. The checked Popper version's ordinary example/bias path represents positive body literals; it does not provide a drop-in equivalent of the ILASP `not wall` mode declaration. The implemented adapter constructs an exact finite per-context `open` relation as the complement of the local wall facts over the supplied cell range, then projects `open` back to `not wall` in the planner rule. This is a controlled semantic compilation, not a claim that Popper's syntax natively supports ILASP negation.

4. **Planner projection.** The planner expects un-reified rules over `state_at/2`, `action/2`, `adjacent/3`, and `wall/1`. A Popper hypothesis learned over context-reified predicates cannot be inserted directly. The adapter must project it to the planner vocabulary, reject hypotheses that retain context arguments in unsafe positions, and preserve the original transition semantics. This projection must be tested with a real learned hypothesis and a real Clingo plan.

Phase 2 resolved these constraints in `symbolic-rl/lib/popper_backend.py` and verified the projection with a real Clingo plan. The `step_<direction>` restriction and finite `open` complement must remain explicit in any later comparison because they preserve the intended transition semantics but are not a byte-for-byte copy of ILASP's search encoding.

## Phase 1 conclusion and Phase 2 acceptance criteria

Phase 1 confirms the following:

- Experience is collected from the existing py-vgdl maze through the standard environment reset/step interface.
- `induction.generate_pos` creates context-dependent ILASP examples from observed transitions and local walls.
- `induction.run_ILASP` is the only induction call boundary used by the main and transfer loops.
- The learned hypothesis is transformed into a temporal ASP transition model by `py_asp.convert_las_asp` and inserted into the existing Clingo planner.
- The Q-learning baseline and evaluation protocol are independent and must not be changed.
- A direct, global Popper conversion would not preserve the original research question because of ILASP context semantics, action constants, wall negation, and planner projection.

Phase 2 is ready to begin with these concrete acceptance tests:

1. Compile one existing `output.las`-style transition task into a context-safe Popper task without changing its examples or intended bias.
2. Run the pinned Popper toolchain and save the learned hypothesis in an inspectable form.
3. Convert that hypothesis to planner syntax with explicit validation and no silent rule loss.
4. Load it into the existing `clingo.lp`, obtain a valid action plan, execute at least one complete maze episode, and log the induction and planning artifacts.
5. Keep the original ILASP path available for side-by-side reference and record any semantic limitation before starting the multi-seed comparison.

## Source links

- [Original README](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/README.md)
- [Authoritative report PDF](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/report.pdf)
- [Configuration](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/config.py)
- [Main ILP(RL) loop](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/main.py)
- [Q-learning baseline](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/q_learning.py)
- [Transfer-learning loop](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/tl.py)
- [Known-goal loop](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/noTL_goal.py)
- [Induction and ILASP invocation](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/lib/induction.py)
- [ASP predicate and hypothesis conversion](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/lib/py_asp.py)
- [Clingo planner](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/lib/abduction.py)
- [ILASP bias and background](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/las_base.las)
- [Transfer examples file](/home/tasin/Documents/For%20Thesis/Final%20Thesis/symbolic-rl/tl_pos.las)
