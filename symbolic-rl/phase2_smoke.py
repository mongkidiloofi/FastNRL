"""Phase 2 smoke test: original maze -> Popper -> existing ASP planner.

This intentionally stays separate from main.py and q_learning.py.  It collects
ordinary four-action transitions from the original ``aaa_small`` py-vgdl maze,
learns once with the Popper backend, inserts the validated projection into the
existing planner skeleton, and executes/replans until the goal or a bounded
step limit is reached.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _install_legacy_runtime_shims():
    """Return gym after the narrowly scoped py-vgdl/Gym compatibility shims."""

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for import_root in (ROOT / "py-vgdl", ROOT / "gym_vgdl"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    import numpy as np

    # Gym 0.26's passive checker refers to the removed NumPy 1.x alias.
    if not hasattr(np, "bool8"):
        np.bool8 = bool

    import pygame
    import pygame.locals as pygame_locals

    # py-vgdl indexes its key-state array with the pygame 1.x key values.
    for name, value in {
        "K_UP": 273,
        "K_DOWN": 274,
        "K_LEFT": 276,
        "K_RIGHT": 275,
        "K_SPACE": 32,
    }.items():
        setattr(pygame, name, value)
        setattr(pygame_locals, name, value)

    import gym
    from gym.envs import registration

    original_register = registration.register

    def register_legacy_compatible(*args, **kwargs):
        if "timestep_limit" in kwargs:
            kwargs["max_episode_steps"] = kwargs.pop("timestep_limit")
        return original_register(*args, **kwargs)

    registration.register = register_legacy_compatible
    import gym_vgdl  # noqa: F401 - importing registers the sample games
    return gym


def _position(env) -> Tuple[int, int]:
    avatars = env.game.getSprites("avatar")
    if avatars:
        sprite = avatars[0]
        return sprite.rect.left // env.game.block_size, sprite.rect.top // env.game.block_size
    raise RuntimeError("the maze avatar no longer exists")


def _goal_position(env) -> Tuple[int, int]:
    goals = env.game.getSprites("goal")
    if not goals:
        raise RuntimeError("the maze has no goal sprite")
    goal = goals[0]
    return goal.rect.left // env.game.block_size, goal.rect.top // env.game.block_size


def _walls(env) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        sorted(
            (sprite.rect.left // env.game.block_size, sprite.rect.top // env.game.block_size)
            for sprite in env.game.getSprites("wall")
        )
    )


def _reset(env) -> None:
    env.reset()
    # VGDLEnv requires one render call before game.tick; rgb_array is headless.
    env.render(mode="rgb_array")


def _replay(env, path: Sequence[int]) -> Tuple[int, int]:
    _reset(env)
    for action in path:
        _state, _reward, done, _info = env.step(action)
        if done:
            raise RuntimeError("a replay path reached the terminal state too early")
    return _position(env)


def _format_atom_set(states: Iterable[Tuple[int, int]]) -> str:
    return ",".join(f"state_after(({x},{y}))" for x, y in states)


def collect_experience(env, output_path: Path) -> Dict[str, object]:
    """Exhaustively collect reachable local transitions as ILASP-style CDPIs."""

    _reset(env)
    start = _position(env)
    goal = _goal_position(env)
    max_x = env.game.screensize[0] // env.game.block_size - 1
    max_y = env.game.screensize[1] // env.game.block_size - 1
    wall_positions = _walls(env)

    queue: List[Tuple[int, ...]] = [()]
    seen = {start}
    examples: List[str] = [f"cell((0..{max_x},0..{max_y}))."]
    transitions = 0
    terminal_transitions = 0

    while queue:
        path = queue.pop(0)
        before = _replay(env, path)
        nearby_walls = [
            (x, y)
            for x, y in wall_positions
            if abs(x - before[0]) + abs(y - before[1]) == 1
        ]

        for action_index, action_name in enumerate(ACTION_NAMES):
            _replay(env, path)
            observed_before = _position(env)
            _state, _reward, done, _info = env.step(action_index)
            try:
                after = _position(env)
            except RuntimeError:
                # The goal interaction can remove the avatar in this old engine.
                # A terminal transition into the known goal is still unambiguous.
                if done:
                    after = goal
                else:
                    raise

            candidates = [
                (observed_before[0] + 1, observed_before[1]),
                (observed_before[0], observed_before[1] + 1),
                (observed_before[0] - 1, observed_before[1]),
                (observed_before[0], observed_before[1] - 1),
                observed_before,
            ]
            exclusions = [candidate for candidate in candidates if candidate != after]
            context = (
                f"state_before(({observed_before[0]},{observed_before[1]})). "
                f"action({action_name}). "
                + " ".join(f"wall(({x},{y}))." for x, y in nearby_walls)
            )
            examples.append(
                f"#pos({{{_format_atom_set([after])}}},"
                f"{{{_format_atom_set(exclusions)}}},"
                f"{{{context}}})."
            )
            transitions += 1
            terminal_transitions += int(done)

            if not done and after not in seen:
                seen.add(after)
                queue.append(path + (action_index,))

    output_path.write_text("\n".join(examples) + "\n")
    return {
        "start": start,
        "goal": goal,
        "max_x": max_x,
        "max_y": max_y,
        "reachable_states": len(seen),
        "transitions": transitions,
        "terminal_transitions": terminal_transitions,
        "walls": wall_positions,
    }


def _write_planner(
    env,
    planner_path: Path,
    hypothesis: str,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    known_walls: Iterable[Tuple[int, int]],
    max_x: int,
    max_y: int,
    horizon: int,
) -> None:
    # Use the original planner construction and marker functions.  Only the
    # Clingo executable call is localised below because this environment has the
    # Python clingo module but not a system-level clingo5 binary.
    abduction = _original_abduction()
    import config as cf

    cf.CLINGOFILE = str(planner_path)
    cf.TIME_RANGE = horizon
    planner_path.unlink(missing_ok=True)
    abduction.make_lp_base(f"cell((0..{max_x},0..{max_y})).\n")
    abduction.add_hypothesis(hypothesis)
    abduction.add_start_state(start)
    abduction.add_goal_state(goal)
    for wall in sorted(set(known_walls)):
        with planner_path.open("a") as planner:
            planner.write(f"wall({wall}).\n")


def _original_abduction():
    """Load the original planner without importing its unused plotting binary."""

    if "lib.plotting" not in sys.modules:
        sys.modules["lib.plotting"] = types.ModuleType("lib.plotting")
    from lib import abduction

    return abduction


def _run_clingo(planner_path: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "clingo",
        "--opt-strat=usc,stratify",
        "-n",
        "0",
        str(planner_path),
        "--opt-mode=opt",
        "--outf=2",
    ]
    output = subprocess.check_output(command, text=True)
    result = json.loads(output)
    if result["Result"] == "UNSATISFIABLE":
        return []
    witnesses = result["Call"][0]["Witnesses"]
    return witnesses[-1]["Value"]


def run_planner_loop(env, metadata: Dict[str, object], hypothesis: str, output_dir: Path) -> Dict[str, object]:
    abduction = _original_abduction()

    start = tuple(metadata["start"])
    goal = tuple(metadata["goal"])
    max_x = int(metadata["max_x"])
    max_y = int(metadata["max_y"])
    all_walls = set(tuple(wall) for wall in metadata["walls"])
    known_walls = set()
    current = start
    actions_taken: List[str] = []
    planner_paths: List[str] = []
    reached_goal = False

    # Experience collection leaves the legacy engine at its final replayed
    # state.  The execution loop is a fresh episode, as in the original code.
    _reset(env)

    for step in range(40):
        known_walls.update(
            wall
            for wall in all_walls
            if abs(wall[0] - current[0]) + abs(wall[1] - current[1]) == 1
        )
        planner_path = output_dir / f"clingo_step_{step:02d}.lp"
        _write_planner(
            env,
            planner_path,
            hypothesis,
            current,
            goal,
            known_walls,
            max_x,
            max_y,
            horizon=40,
        )
        planner_paths.append(str(planner_path))
        answer_set = _run_clingo(planner_path)
        _states, planned_actions = abduction.sort_planning(answer_set)
        # The original planner declares time(0..T) but seeds the current
        # state at time 1.  Ignore the unconstrained action at time 0.
        planned_actions = [item for item in planned_actions if item[0] >= 1]
        if not planned_actions:
            break
        action_name = planned_actions[0][1]
        if action_name not in ACTION_NAMES:
            raise RuntimeError(f"planner returned unsupported action {action_name!r}")
        action_index = ACTION_NAMES.index(action_name)
        before = _position(env)
        _state, _reward, done, _info = env.step(action_index)
        try:
            current = _position(env)
        except RuntimeError:
            current = goal if done else before
        actions_taken.append(action_name)
        if done or current == goal:
            reached_goal = True
            break

    return {
        "reached_goal": reached_goal,
        "actions": actions_taken,
        "steps": len(actions_taken),
        "final_position": current,
        "planner_files": planner_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/symbolic-rl-phase2")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gym = _install_legacy_runtime_shims()
    env = gym.make("vgdl_aaa_small-v0", disable_env_checker=True).unwrapped
    metadata = collect_experience(env, output_dir / "experience.las")

    from lib.popper_backend import (
        convert_popper_hypothesis_to_planner,
        learn_transition_hypothesis,
    )

    task_dir = output_dir / "popper_task"
    contextual = learn_transition_hypothesis(
        output_dir / "experience.las",
        output_dir / "experience.las",
        ROOT / "las_base.las",
        popper_root=ROOT.parent / "Popper",
        output_dir=task_dir,
        timeout=120,
    )
    planner_hypothesis = convert_popper_hypothesis_to_planner(contextual)
    (output_dir / "planner_hypothesis.lp").write_text(planner_hypothesis)
    loop = run_planner_loop(env, metadata, planner_hypothesis, output_dir)
    result = {
        "environment": "vgdl_aaa_small-v0",
        "backend": "popper",
        "popper_root": str((ROOT.parent / "Popper").resolve()),
        "metadata": metadata,
        "contextual_hypothesis": contextual,
        "planner_hypothesis": planner_hypothesis,
        "loop": loop,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=list) + "\n")
    print(json.dumps({"metadata": metadata, "loop": loop}, indent=2, default=list))
    return 0 if loop["reached_goal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
