"""Phase 3 experiment entry point.

The original ILASP executable is retained as a reference backend, but Phase 0
showed that the real symbolic-rl ILASP tasks cannot currently be rerun.  The
Popper sweep therefore reports a reference-only ILASP status instead of
inventing replacement numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


WORKSPACE = Path(__file__).resolve().parent
SYMBOLIC_RL = WORKSPACE / "symbolic-rl"
sys.path.insert(0, str(SYMBOLIC_RL))

import phase2_smoke as phase2


ACTION_NAMES = phase2.ACTION_NAMES
BASE_BIAS = SYMBOLIC_RL / "las_base.las"


def make_env(gym, game: str):
    return gym.make(f"vgdl_{game}-v0", disable_env_checker=True).unwrapped


def _context_example(before, action_name, after, nearby_walls) -> str:
    candidates = [
        (before[0] + 1, before[1]),
        (before[0], before[1] + 1),
        (before[0] - 1, before[1]),
        (before[0], before[1] - 1),
        before,
    ]
    exclusions = [candidate for candidate in candidates if candidate != after]
    inc = f"state_after(({after[0]},{after[1]}))"
    exc = ",".join(f"state_after(({x},{y}))" for x, y in exclusions)
    walls = " ".join(f"wall(({x},{y}))." for x, y in nearby_walls)
    context = f"state_before(({before[0]},{before[1]})). action({action_name}). {walls}"
    return f"#pos({{{inc}}},{{{exc}}},{{{context}}})."


def collect_online_examples(env, rng: random.Random, episodes: int, steps: int) -> Tuple[str, Dict[str, object]]:
    """Collect transitions using the original four-action exploration semantics."""

    phase2._reset(env)
    start = phase2._position(env)
    goal = phase2._goal_position(env)
    max_x = env.game.screensize[0] // env.game.block_size - 1
    max_y = env.game.screensize[1] // env.game.block_size - 1
    walls = phase2._walls(env)
    lines = [f"cell((0..{max_x},0..{max_y}))."]
    training_rewards: List[float] = []
    training_successes = 0
    transitions = 0

    for _episode in range(episodes):
        phase2._reset(env)
        episode_reward = 0.0
        done = False
        for _step in range(steps):
            before = phase2._position(env)
            nearby = [
                wall
                for wall in walls
                if abs(wall[0] - before[0]) + abs(wall[1] - before[1]) == 1
            ]
            action_index = rng.randrange(4)
            action_name = ACTION_NAMES[action_index]
            _obs, _reward, done, _info = env.step(action_index)
            try:
                after = phase2._position(env)
            except RuntimeError:
                after = goal if done else before
            lines.append(_context_example(before, action_name, after, nearby))
            transitions += 1
            episode_reward += 10 if done else -1
            if done:
                training_successes += 1
                break
        training_rewards.append(episode_reward)

    metadata = {
        "start": start,
        "goal": goal,
        "max_x": max_x,
        "max_y": max_y,
        "walls": walls,
        "collection_episodes": episodes,
        "collection_steps": steps,
        "transitions": transitions,
        "training_successes": training_successes,
        "training_rewards": training_rewards,
    }
    return "\n".join(lines) + "\n", metadata


def maze_metadata(env) -> Dict[str, object]:
    phase2._reset(env)
    return {
        "start": phase2._position(env),
        "goal": phase2._goal_position(env),
        "max_x": env.game.screensize[0] // env.game.block_size - 1,
        "max_y": env.game.screensize[1] // env.game.block_size - 1,
        "walls": phase2._walls(env),
    }


def _greedy_action(q_values: Sequence[float]) -> int:
    return max(range(len(q_values)), key=lambda index: (q_values[index], -index))


def _epsilon_action(q_values: Sequence[float], rng: random.Random, epsilon: float) -> int:
    best = _greedy_action(q_values)
    probabilities = [epsilon / 4.0] * 4
    probabilities[best] += 1.0 - epsilon
    draw = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            return action
    return 3


def _split_logic_terms(text: str) -> List[str]:
    """Split a Prolog/ASP argument or body list at top-level commas."""

    parts: List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _logic_term(text: str):
    text = text.strip()
    if text and text[0].isupper() and text.replace("_", "").isalnum():
        return ("var", text)
    cell = text.strip()
    if cell.startswith("cell(") and cell.endswith(")"):
        values = _split_logic_terms(cell[5:-1])
        if len(values) == 2 and all(value.lstrip("-").isdigit() for value in values):
            return ("cell", int(values[0]), int(values[1]))
    return ("atom", text)


def _logic_atom(text: str):
    text = text.strip().rstrip(".")
    negative = text.startswith("not ")
    if negative:
        text = text[4:].strip()
    opening = text.find("(")
    if opening < 1 or not text.endswith(")"):
        raise ValueError(f"unsupported hypothesis literal: {text!r}")
    predicate = text[:opening].strip()
    arguments = tuple(_logic_term(value) for value in _split_logic_terms(text[opening + 1:-1]))
    return negative, predicate, arguments


def _parse_contextual_hypothesis(hypothesis: str):
    rules = []
    for raw_line in hypothesis.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("%%"):
            continue
        line = line.rstrip(".")
        if ":-" not in line:
            continue
        head_text, body_text = line.split(":-", 1)
        _negative, head_predicate, head_arguments = _logic_atom(head_text)
        body = tuple(_logic_atom(literal) for literal in _split_logic_terms(body_text))
        rules.append((head_predicate, head_arguments, body))
    return tuple(rules)


def _logic_unify(pattern, value, bindings: Dict[str, object]) -> bool:
    if isinstance(pattern, tuple) and pattern and pattern[0] == "var":
        name = pattern[1]
        if name in bindings:
            return _logic_unify(bindings[name], value, bindings)
        bindings[name] = value
        return True
    if isinstance(pattern, tuple) and isinstance(value, tuple):
        if pattern[0] != value[0] or len(pattern) != len(value):
            return False
        return all(_logic_unify(left, right, bindings) for left, right in zip(pattern[1:], value[1:]))
    return pattern == value


def _hypothesis_entails(rules, facts, query) -> bool:
    query_predicate, query_arguments = query
    by_predicate = defaultdict(list)
    for predicate, arguments in facts:
        by_predicate[predicate].append(arguments)

    for head_predicate, head_arguments, body in rules:
        if head_predicate != query_predicate:
            continue
        bindings: Dict[str, object] = {}
        if not all(_logic_unify(pattern, value, bindings) for pattern, value in zip(head_arguments, query_arguments)):
            continue
        environments = [bindings]
        for negative, predicate, arguments in body:
            next_environments = []
            for environment in environments:
                matches = []
                for fact_arguments in by_predicate.get(predicate, []):
                    candidate = dict(environment)
                    if all(_logic_unify(pattern, value, candidate) for pattern, value in zip(arguments, fact_arguments)):
                        matches.append(candidate)
                if negative:
                    if not matches:
                        next_environments.append(environment)
                else:
                    next_environments.extend(matches)
            environments = next_environments
            if not environments:
                break
        if environments:
            return True
    return False


def hypothesis_covers_transition(
    contextual_hypothesis: str,
    before: Tuple[int, int],
    action_name: str,
    after: Tuple[int, int],
    nearby_walls: Iterable[Tuple[int, int]],
    max_x: int,
    max_y: int,
    context_name: str,
) -> bool:
    """Check the current contextual hypothesis against one ILASP-style CDPI."""

    rules = _parse_contextual_hypothesis(contextual_hypothesis)
    context = ("atom", context_name)
    facts = [("state_before", (context, ("cell", before[0], before[1])))]
    walls = set(nearby_walls)
    for direction, (dx, dy) in phase2.ACTION_DELTAS.items():
        target = (before[0] + dx, before[1] + dy)
        if not (0 <= target[0] <= max_x and 0 <= target[1] <= max_y):
            continue
        target_term = ("cell", target[0], target[1])
        source_term = ("cell", before[0], before[1])
        if direction == action_name:
            facts.append((f"step_{direction}", (context, target_term, source_term)))
        facts.append(("wall" if target in walls else "open", (context, target_term)))
    facts.extend(("wall", (context, ("cell", x, y))) for x, y in walls)

    def entails(position: Tuple[int, int]) -> bool:
        return _hypothesis_entails(
            rules,
            facts,
            ("transition", (context, ("cell", position[0], position[1]))),
        )

    candidates = [
        (before[0] + 1, before[1]),
        (before[0], before[1] + 1),
        (before[0] - 1, before[1]),
        (before[0], before[1] - 1),
        before,
    ]
    exclusions = [candidate for candidate in candidates if candidate != after]
    return entails(after) and not any(entails(candidate) for candidate in exclusions)


def evaluate_q(env, q: Dict[Tuple[int, int], List[float]], episodes: int, steps: int) -> Dict[str, object]:
    rewards: List[float] = []
    successes: List[int] = []
    lengths: List[int] = []
    for _ in range(episodes):
        phase2._reset(env)
        total = 0.0
        success = 0
        length = steps
        for step in range(steps):
            state = phase2._position(env)
            action = _greedy_action(q.get(state, [0.0] * 4))
            _obs, _reward, done, _info = env.step(action)
            total += 10 if done else -1
            if done:
                success = 1
                length = step + 1
                break
        rewards.append(total)
        successes.append(success)
        lengths.append(length)
    return {
        "evaluation_mean_reward": mean(rewards),
        "evaluation_reward_spread": pstdev(rewards),
        "evaluation_success_rate": mean(successes),
        "evaluation_mean_steps": mean(lengths),
        "evaluation_rewards": rewards,
        "evaluation_successes": successes,
    }


def _episodes_to_optimal_policy(evaluation: Sequence[Dict[str, object]]) -> int:
    """Return the first greedy evaluation matching the final Q policy."""

    if not evaluation:
        return 0
    final = evaluation[-1]
    for index, result in enumerate(evaluation):
        if (
            result["evaluation_success_rate"] == final["evaluation_success_rate"]
            and result["evaluation_mean_steps"] == final["evaluation_mean_steps"]
        ):
            return index + 1
    return 0


def run_q_learning(env, seed: int, episodes: int, steps: int) -> Dict[str, object]:
    rng = random.Random(seed)
    q: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0] * 4)
    training_rewards: List[float] = []
    training_successes: List[int] = []
    training_by_episode: List[Dict[str, object]] = []
    evaluation: List[Dict[str, object]] = []
    start_time = time.perf_counter()

    for _episode in range(episodes):
        phase2._reset(env)
        total = 0.0
        success = 0
        episode_steps = 0
        for _step in range(steps):
            state = phase2._position(env)
            action = _epsilon_action(q[state], rng, epsilon=0.1)
            _obs, _reward, done, _info = env.step(action)
            next_state = phase2._position(env) if not done else state
            reward = 10 if done else -1
            total += reward
            episode_steps = _step + 1
            best_next = max(q[next_state])
            q[state][action] += 0.5 * (reward + best_next - q[state][action])
            if done:
                success = 1
                break
        training_rewards.append(total)
        training_successes.append(success)
        training_by_episode.append(
            {
                "episode": _episode + 1,
                "reward": total,
                "success": success,
                "steps": episode_steps,
            }
        )
        evaluation.append(evaluate_q(env, q, 1, steps))

    final = evaluation[-1]
    return {
        "backend": "q_learning",
        "seed": seed,
        "episodes": episodes,
        "steps_per_episode": steps,
        "training_mean_reward": mean(training_rewards),
        "training_success_rate": mean(training_successes),
        "training_rewards": training_rewards,
        "training_successes": training_successes,
        "training_by_episode": training_by_episode,
        "evaluation_by_episode": evaluation,
        "episodes_to_optimal_policy": _episodes_to_optimal_policy(evaluation),
        "final_evaluation": final,
        "evaluation_mean_reward": final["evaluation_mean_reward"],
        "evaluation_reward_spread": final["evaluation_reward_spread"],
        "evaluation_success_rate": final["evaluation_success_rate"],
        "evaluation_mean_steps": final["evaluation_mean_steps"],
        "runtime_seconds": time.perf_counter() - start_time,
    }


def plan_with_all_walls(env, metadata: Dict[str, object], planner_hypothesis: str, output_dir: Path) -> List[int]:
    planner_path = output_dir / "evaluation_planner.lp"
    phase2._write_planner(
        env,
        planner_path,
        planner_hypothesis,
        tuple(metadata["start"]),
        tuple(metadata["goal"]),
        [tuple(wall) for wall in metadata["walls"]],
        int(metadata["max_x"]),
        int(metadata["max_y"]),
        horizon=250,
    )
    answer_set = phase2._run_clingo(planner_path)
    abduction = phase2._original_abduction()
    _states, actions = abduction.sort_planning(answer_set)
    return [ACTION_NAMES.index(action) for timestamp, action in actions if timestamp >= 1 and action in ACTION_NAMES]


def evaluate_fixed_plan(env, action_plan: Sequence[int], episodes: int, steps: int) -> Dict[str, object]:
    rewards: List[float] = []
    successes: List[int] = []
    lengths: List[int] = []
    for _ in range(episodes):
        phase2._reset(env)
        total = 0.0
        success = 0
        length = steps
        for step, action in enumerate(action_plan[:steps]):
            _obs, _reward, done, _info = env.step(action)
            total += 10 if done else -1
            if done:
                success = 1
                length = step + 1
                break
        rewards.append(total)
        successes.append(success)
        lengths.append(length)
    return {
        "evaluation_mean_reward": mean(rewards),
        "evaluation_reward_spread": pstdev(rewards),
        "evaluation_success_rate": mean(successes),
        "evaluation_mean_steps": mean(lengths),
        "evaluation_rewards": rewards,
        "evaluation_successes": successes,
    }


def run_popper(
    env,
    seed: int,
    episodes: int,
    steps: int,
    output_dir: Path,
    collection_episodes: int,
    evaluation_env=None,
    evaluation_metadata: Dict[str, object] = None,
) -> Dict[str, object]:
    rng = random.Random(seed)
    start_time = time.perf_counter()
    phase2._reset(env)
    start = phase2._position(env)
    goal = phase2._goal_position(env)
    max_x = env.game.screensize[0] // env.game.block_size - 1
    max_y = env.game.screensize[1] // env.game.block_size - 1
    walls = phase2._walls(env)
    experience_lines = [f"cell((0..{max_x},0..{max_y}))."]
    experience_path = output_dir / "experience.las"
    task_dir = output_dir / "popper_task"

    from lib.popper_backend import convert_popper_hypothesis_to_planner, learn_transition_hypothesis

    contextual = ""
    planner_hypothesis = ""
    induction_seconds = 0.0
    induction_episodes: List[int] = []
    induction_calls = 0
    projection_error = None
    training_rewards: List[float] = []
    training_successes: List[int] = []
    training_by_episode: List[Dict[str, object]] = []
    transitions = 0

    def induce() -> None:
        nonlocal contextual, planner_hypothesis, induction_seconds, induction_calls, projection_error
        experience_path.write_text("\n".join(experience_lines) + "\n")
        induction_start = time.perf_counter()
        contextual = learn_transition_hypothesis(
            experience_path,
            experience_path,
            BASE_BIAS,
            popper_root=WORKSPACE / "Popper",
            output_dir=task_dir,
            timeout=120,
        )
        induction_seconds += time.perf_counter() - induction_start
        induction_calls += 1
        planner_hypothesis = ""
        try:
            planner_hypothesis = convert_popper_hypothesis_to_planner(contextual)
            projection_error = None
        except Exception as exc:
            # Popper can return a consistent partial hypothesis while more
            # transition contexts are still being collected.  Keep that
            # hypothesis for the next coverage check; projection is required
            # only for the final planner evaluation.
            projection_error = f"{type(exc).__name__}: {exc}"

    observed_actions = set()
    for episode in range(1, collection_episodes + 1):
        phase2._reset(env)
        episode_reward = 0.0
        episode_success = 0
        episode_steps = 0
        done = False
        for _step in range(steps):
            before = phase2._position(env)
            nearby = [
                wall
                for wall in walls
                if abs(wall[0] - before[0]) + abs(wall[1] - before[1]) == 1
            ]
            action_index = rng.randrange(4)
            action_name = ACTION_NAMES[action_index]
            observed_actions.add(action_name)
            _obs, _reward, done, _info = env.step(action_index)
            try:
                after = phase2._position(env)
            except RuntimeError:
                after = goal if done else before

            context_name = f"c{transitions}"
            experience_lines.append(_context_example(before, action_name, after, nearby))
            transitions += 1
            episode_steps = _step + 1

            covered = False
            if contextual:
                covered = hypothesis_covers_transition(
                    contextual,
                    before,
                    action_name,
                    after,
                    nearby,
                    max_x,
                    max_y,
                    context_name,
                )
            elif len(observed_actions) < len(ACTION_NAMES):
                # The unchanged Popper adapter needs at least one example for
                # every action predicate before SWI can load its BK safely.
                covered = True
            if not covered:
                induce()
                induction_episodes.append(episode)

            episode_reward += 10 if done else -1
            if done:
                episode_success = 1
                break

        training_rewards.append(episode_reward)
        training_successes.append(episode_success)
        training_by_episode.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "success": episode_success,
                "steps": episode_steps,
            }
        )

    experience_path.write_text("\n".join(experience_lines) + "\n")
    if not contextual:
        induce()
        induction_episodes.append(collection_episodes)
    if not planner_hypothesis:
        raise RuntimeError(
            "final Popper hypothesis could not be projected to the planner"
            + (f": {projection_error}" if projection_error else "")
        )
    (output_dir / "planner_hypothesis.lp").write_text(planner_hypothesis)
    if evaluation_env is None:
        evaluation_env = env
    if evaluation_metadata is None:
        evaluation_metadata = {
            "start": start,
            "goal": goal,
            "max_x": max_x,
            "max_y": max_y,
            "walls": walls,
        }
    action_plan = plan_with_all_walls(evaluation_env, evaluation_metadata, planner_hypothesis, output_dir)
    evaluation = evaluate_fixed_plan(evaluation_env, action_plan, episodes, steps)
    return {
        "backend": "popper",
        "seed": seed,
        "episodes": episodes,
        "steps_per_episode": steps,
        "collection_episodes": collection_episodes,
        "collection_transitions": transitions,
        "collection_success_rate": mean(training_successes),
        "training_rewards": training_rewards,
        "training_successes": training_successes,
        "training_by_episode": training_by_episode,
        "episodes_to_convergence": induction_episodes[-1],
        "induction_episodes": induction_episodes,
        "induction_calls": induction_calls,
        "induction_seconds": induction_seconds,
        "hypothesis_rules": len([line for line in contextual.splitlines() if line.strip()]),
        "hypothesis_size_chars": len(contextual),
        "action_plan": action_plan,
        "planner_plan_steps": len(action_plan),
        "contextual_hypothesis": contextual,
        "planner_hypothesis": planner_hypothesis,
        **evaluation,
        "runtime_seconds": time.perf_counter() - start_time,
        "protocol_note": "Popper collects transitions incrementally and re-induces only when the current hypothesis fails the observed CDPI coverage check; this remains a Popper replacement, not a claim of exact ILASP call-for-call equivalence.",
    }


def run_ilasp_reference(seed: int, experiment: str) -> Dict[str, object]:
    return {
        "backend": "ilasp",
        "seed": seed,
        "experiment": experiment,
        "status": "reference_only_unavailable",
        "reason": "Phase 0: ILASP2i launches, but real symbolic-rl context-dependent tasks fail with Poco::WriteFileException in this environment.",
        "published_reference": "symbolic-rl/report.pdf",
    }


def _run_isolated_popper_sweep(args, root_output: Path) -> List[Dict[str, object]]:
    """Run each Popper seed in a fresh process because SWI consults globally."""

    def run_child(seed: int) -> None:
        child_env = dict(os.environ)
        child_env["PHASE3_POPPER_CHILD"] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            "popper",
            "--experiment",
            args.experiment,
            "--seed",
            str(seed),
            "--runs",
            "1",
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--collection-episodes",
            str(args.collection_episodes if args.collection_episodes is not None else args.episodes),
            "--game",
            args.game,
            "--transfer-game",
            args.transfer_game,
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ]
        subprocess.run(command, check=False, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor(max_workers=min(6, args.runs)) as executor:
        futures = [executor.submit(run_child, args.seed + offset) for offset in range(args.runs)]
        for future in as_completed(futures):
            future.result()
    rows = []
    for offset in range(args.runs):
        result_path = root_output / f"seed_{args.seed + offset}" / "result.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text()))
        else:
            rows.append({"backend": "popper", "seed": args.seed + offset, "status": "failed_no_result"})
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    numeric = {}
    for key in (
        "evaluation_success_rate",
        "evaluation_mean_reward",
        "evaluation_mean_steps",
        "induction_seconds",
        "planner_plan_steps",
        "induction_calls",
        "episodes_to_convergence",
        "episodes_to_optimal_policy",
    ):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            numeric[key] = {"mean": mean(values), "spread": pstdev(values), "n": len(values)}
    return {"runs": len(rows), "metrics": numeric}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ilasp", "popper", "q_learning"), required=True)
    parser.add_argument("--experiment", choices=("baseline", "transfer"), default="baseline")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--collection-episodes",
        type=int,
        default=None,
        help="Popper collection episodes; defaults to the main episode budget",
    )
    parser.add_argument("--game", default="aaa_small")
    parser.add_argument("--transfer-game", default="aaa_medium")
    parser.add_argument("--output-dir", default=str(WORKSPACE / "phase3_results"))
    args = parser.parse_args()
    collection_episodes = args.episodes if args.collection_episodes is None else args.collection_episodes

    root_output = Path(args.output_dir).resolve() / args.experiment / args.backend
    root_output.mkdir(parents=True, exist_ok=True)

    if args.backend == "popper" and args.runs > 1 and os.environ.get("PHASE3_POPPER_CHILD") != "1":
        rows = _run_isolated_popper_sweep(args, root_output)
        summary = summarize(rows)
        summary["backend"] = args.backend
        summary["experiment"] = args.experiment
        summary["seed_start"] = args.seed
        summary["runs"] = args.runs
        (root_output / "summary.json").write_text(json.dumps(summary, indent=2, default=list) + "\n")
        fields = sorted({key for row in rows for key, value in row.items() if isinstance(value, (str, int, float, bool))})
        with (root_output / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})
        print(json.dumps(summary, indent=2, default=list))
        return 0

    gym = phase2._install_legacy_runtime_shims()
    rows: List[Dict[str, object]] = []

    for offset in range(args.runs):
        seed = args.seed + offset
        run_output = root_output / f"seed_{seed}"
        run_output.mkdir(parents=True, exist_ok=True)
        if args.backend == "ilasp":
            row = run_ilasp_reference(seed, args.experiment)
        else:
            train_game = args.game
            eval_game = args.transfer_game if args.experiment == "transfer" else train_game
            env = make_env(gym, train_game)
            evaluation_env = make_env(gym, eval_game) if eval_game != train_game else env
            if args.backend == "q_learning":
                row = run_q_learning(env, seed, args.episodes, args.steps)
                if args.experiment == "transfer":
                    row["status"] = "training_only_no_cross_maze_q_baseline"
            else:
                try:
                    row = run_popper(
                        env,
                        seed,
                        args.episodes,
                        args.steps,
                        run_output,
                        collection_episodes,
                        evaluation_env=evaluation_env,
                        evaluation_metadata=maze_metadata(evaluation_env),
                    )
                except Exception as exc:
                    row = {
                        "backend": "popper",
                        "seed": seed,
                        "status": "failed",
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
            row["experiment"] = args.experiment
            row["training_game"] = train_game
            row["evaluation_game"] = eval_game
            env.close()
            if evaluation_env is not env:
                evaluation_env.close()
        (run_output / "result.json").write_text(json.dumps(row, indent=2, default=list) + "\n")
        rows.append(row)

    summary = summarize(rows)
    summary["backend"] = args.backend
    summary["experiment"] = args.experiment
    summary["seed_start"] = args.seed
    summary["runs"] = args.runs
    (root_output / "summary.json").write_text(json.dumps(summary, indent=2, default=list) + "\n")
    fields = sorted({key for row in rows for key, value in row.items() if isinstance(value, (str, int, float, bool))})
    with (root_output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(json.dumps(summary, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
