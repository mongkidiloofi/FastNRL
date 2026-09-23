"""Popper backend for the original symbolic-rl transition task.

The original project uses ILASP context-dependent partial interpretations.  This
module compiles those examples into a context-reified Popper task, learns a
transition relation, and validates a projection back to the vocabulary used by
the existing Clingo planner.

The compiler is deliberately narrow: it supports the ordinary four-action maze
task represented by ``las_base.las``.  Link/teleport modes and arbitrary ILASP
syntax are rejected instead of being silently approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple, Union


class PopperBackendError(RuntimeError):
    """Raised when the ILASP task cannot be preserved by this adapter."""


Coord = Tuple[int, int]
Source = Union[str, Path]

_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class ILASPExample:
    """The supported three-part form of an ILASP ``#pos`` example."""

    inclusions: Tuple[Coord, ...]
    exclusions: Tuple[Coord, ...]
    before: Coord
    action: str
    walls: Tuple[Coord, ...]


def _read_source(source: Source) -> str:
    path = Path(source)
    try:
        if path.exists():
            return path.read_text()
    except OSError:
        # A task string can be longer than the platform's maximum filename
        # length.  In that case it is necessarily content, not a path.
        pass
    return str(source)


def _split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split text while ignoring separators inside parentheses/braces."""

    parts: List[str] = []
    start = 0
    paren = 0
    brace = 0
    for index, char in enumerate(text):
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == separator and paren == 0 and brace == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _find_pos_bodies(text: str) -> List[str]:
    bodies: List[str] = []
    cursor = 0
    marker = "#pos("
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return bodies
        index = start + len(marker)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        if depth:
            raise PopperBackendError("unterminated #pos example")
        bodies.append(text[start + len(marker): index - 1])
        cursor = index


def _parse_coords(text: str, predicate: str) -> Tuple[Coord, ...]:
    pattern = re.compile(
        rf"{predicate}\s*\(\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\)"
    )
    return tuple((int(x), int(y)) for x, y in pattern.findall(text))


def _parse_action(text: str) -> str:
    actions = re.findall(r"action\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", text)
    if len(actions) != 1:
        raise PopperBackendError(f"expected exactly one action in context: {text!r}")
    action = actions[0]
    if action not in _DIRECTIONS:
        raise PopperBackendError(
            f"action {action!r} is outside the ordinary four-action maze task"
        )
    return action


def parse_ilasp_examples(task: Source) -> Tuple[ILASPExample, ...]:
    """Parse ordinary ILASP CDPIs from a task file or task string."""

    text = _read_source(task)
    examples: List[ILASPExample] = []
    for body in _find_pos_bodies(text):
        fields = _split_top_level(body)
        if len(fields) != 3 or not all(field.startswith("{") and field.endswith("}") for field in fields):
            raise PopperBackendError(
                "only three-set ILASP #pos examples are supported: Einc, Eexc, Context"
            )
        inclusion = _parse_coords(fields[0], "state_after")
        exclusion = _parse_coords(fields[1], "state_after")
        context = fields[2]
        before = _parse_coords(context, "state_before")
        if len(before) != 1:
            raise PopperBackendError("each context must contain exactly one state_before")
        walls = _parse_coords(context, "wall")
        examples.append(
            ILASPExample(
                inclusions=inclusion,
                exclusions=exclusion,
                before=before[0],
                action=_parse_action(context),
                walls=tuple(dict.fromkeys(walls)),
            )
        )
    if not examples:
        raise PopperBackendError("the ILASP task contains no #pos examples")
    return tuple(examples)


def _parse_bounds(*sources: str) -> Tuple[int, int]:
    pattern = re.compile(
        r"cell\s*\(\s*\(\s*0\s*\.\.\s*(\d+)\s*,\s*0\s*\.\.\s*(\d+)\s*\)\s*\)"
    )
    matches = [match for source in sources for match in pattern.findall(source)]
    if not matches:
        raise PopperBackendError("no cell((0..X,0..Y)) range was found in the ILASP task")
    return max(int(x) for x, _ in matches), max(int(y) for _, y in matches)


def _cell(x: int, y: int) -> str:
    return f"cell({x},{y})"


def _popper_bias() -> str:
    lines = [
        "head_pred(transition,2).",
        "body_pred(state_before,2).",
        "body_pred(wall,2).",
        "body_pred(open,2).",
        "type(transition,(context,cell)).",
        "type(state_before,(context,cell)).",
        "type(wall,(context,cell)).",
        "type(open,(context,cell)).",
        "direction(transition,(in,out)).",
        "direction(state_before,(in,out)).",
        "direction(wall,(in,in)).",
        "direction(open,(in,in)).",
    ]
    for action in _DIRECTIONS:
        lines.extend(
            [
                f"body_pred(step_{action},3).",
                f"type(step_{action},(context,cell,cell)).",
                f"direction(step_{action},(in,out,in)).",
            ]
        )
    # The original bias allows eight transition clauses (move/stay for four
    # actions) and at most four body concepts per clause.
    lines.extend(["max_clauses(8).", "max_body(5).", "max_vars(5)."])
    return "\n".join(lines) + "\n"


def _validate_original_bias(bias_text: str) -> None:
    required = [
        "#modeh(state_after(var(cell))).",
        "#modeb(1, state_before(var(cell)), (positive)).",
        "#modeb(1, action(const(action)), (positive)).",
        "#modeb(1, wall(var(cell))).",
        "#constant(action,right).",
        "#constant(action,left).",
        "#constant(action,down).",
        "#constant(action,up).",
    ]
    normalized = re.sub(r"\s+", "", bias_text)
    for declaration in required:
        if re.sub(r"\s+", "", declaration) not in normalized:
            raise PopperBackendError(
                f"the supplied ILASP bias is missing required declaration: {declaration}"
            )


def compile_ilasp_to_popper(
    examples: Source,
    background_knowledge: Source,
    bias: Source,
    output_dir: Source,
) -> Tuple[Path, Tuple[ILASPExample, ...]]:
    """Compile one ordinary ILASP task into Popper's three input files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples_text = _read_source(examples)
    background_text = _read_source(background_knowledge)
    bias_text = _read_source(bias)
    _validate_original_bias(bias_text)
    parsed = parse_ilasp_examples(examples_text)
    max_x, max_y = _parse_bounds(examples_text, background_text)

    bk_lines: List[str] = []
    positive_lines: List[str] = []
    negative_lines: List[str] = []
    for index, example in enumerate(parsed):
        context = f"c{index}"
        before = example.before
        bx, by = before
        bk_lines.append(f"state_before({context},{_cell(bx, by)}).")
        wall_set = set(example.walls)
        for action, (dx, dy) in _DIRECTIONS.items():
            target = (bx + dx, by + dy)
            if not (0 <= target[0] <= max_x and 0 <= target[1] <= max_y):
                continue
            target_term = _cell(*target)
            source_term = _cell(bx, by)
            if action == example.action:
                bk_lines.append(f"step_{action}({context},{target_term},{source_term}).")
            if target in wall_set:
                bk_lines.append(f"wall({context},{target_term}).")
            else:
                bk_lines.append(f"open({context},{target_term}).")
        for wall in sorted(wall_set):
            bk_lines.append(f"wall({context},{_cell(*wall)}).")

        for target in example.inclusions:
            positive_lines.append(f"pos(transition({context},{_cell(*target)})).")
        for target in example.exclusions:
            negative_lines.append(f"neg(transition({context},{_cell(*target)})).")

    bk_path = output / "bk.pl"
    ex_path = output / "exs.pl"
    bias_path = output / "bias.pl"
    bk_path.write_text(":- style_check(-discontiguous).\n" + "\n".join(bk_lines) + "\n")
    ex_path.write_text(":- style_check(-discontiguous).\n" + "\n".join(positive_lines + negative_lines) + "\n")
    bias_path.write_text(_popper_bias())
    (output / "source_ilasp.las").write_text(examples_text)
    (output / "source_bias.las").write_text(bias_text)
    return output, parsed


def _load_popper(popper_root: Optional[Source]):
    if popper_root is None:
        popper_root = Path(__file__).resolve().parents[2] / "Popper"
    root = str(Path(popper_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from popper.loop import learn_solution
        from popper.util import Settings, format_prog, order_prog
    except Exception as exc:  # pragma: no cover - exercised by environment setup
        raise PopperBackendError(
            "Popper could not be imported; configure its pinned SWI/Clingo runtime"
        ) from exc
    return learn_solution, Settings, format_prog, order_prog


def learn_transition_hypothesis(
    examples: Source,
    background_knowledge: Source,
    bias: Source,
    *,
    popper_root: Optional[Source] = None,
    output_dir: Optional[Source] = None,
    timeout: float = 120.0,
) -> str:
    """Learn a context-reified transition hypothesis with Popper.

    The returned text is Popper's ordinary Prolog hypothesis, using the
    contextual predicate ``transition(Context,Cell)``.  Call
    :func:`convert_popper_hypothesis_to_planner` before inserting it into
    ``clingo.lp``.
    """

    keep_dir = Path(output_dir) if output_dir is not None else None
    if keep_dir is not None:
        task_dir = keep_dir
        task_dir.mkdir(parents=True, exist_ok=True)
        return _learn_from_dir(
            examples, background_knowledge, bias, task_dir, popper_root, timeout
        )

    with tempfile.TemporaryDirectory(prefix="symbolic_rl_popper_") as temporary:
        return _learn_from_dir(
            examples,
            background_knowledge,
            bias,
            Path(temporary),
            popper_root,
            timeout,
        )


def _learn_from_dir(
    examples: Source,
    background_knowledge: Source,
    bias: Source,
    task_dir: Path,
    popper_root: Optional[Source],
    timeout: float,
) -> str:
    compile_ilasp_to_popper(examples, background_knowledge, bias, task_dir)
    learn_solution, Settings, format_prog, order_prog = _load_popper(popper_root)
    settings = Settings(
        kbpath=str(task_dir),
        info=False,
        timeout=timeout,
        max_body=5,
        max_vars=5,
        max_rules=8,
        max_literals=48,
        solver="rc2",
    )
    program, score, stats = learn_solution(settings)
    if program is None:
        raise PopperBackendError("Popper found no consistent transition hypothesis")
    hypothesis = format_prog(order_prog(program))
    (task_dir / "hypothesis.pl").write_text(hypothesis + "\n")
    (task_dir / "score.txt").write_text(str(score) + "\n")
    return hypothesis


def _parse_literal(text: str) -> Tuple[str, Tuple[str, ...]]:
    match = re.fullmatch(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*", text)
    if not match:
        raise PopperBackendError(f"cannot parse Popper literal: {text!r}")
    return match.group(1), tuple(_split_top_level(match.group(2)))


def _parse_rule(text: str) -> Tuple[Tuple[str, Tuple[str, ...]], List[Tuple[str, Tuple[str, ...]]]]:
    text = text.strip().rstrip(".").strip()
    if ":-" not in text:
        raise PopperBackendError(f"Popper hypothesis contains a fact, not a rule: {text!r}")
    head_text, body_text = text.split(":-", 1)
    return _parse_literal(head_text), [_parse_literal(x) for x in _split_top_level(body_text)]


def convert_popper_hypothesis_to_planner(hypothesis: str) -> str:
    """Validate and project a Popper hypothesis into the original ASP syntax."""

    rules = [line.strip() for line in hypothesis.splitlines() if line.strip() and not line.strip().startswith("%")]
    if not rules:
        raise PopperBackendError("Popper returned an empty hypothesis")
    converted: List[str] = []
    for rule_text in rules:
        (head_pred, head_args), body = _parse_rule(rule_text)
        if head_pred != "transition" or len(head_args) != 2:
            raise PopperBackendError(f"unsupported Popper head: {head_pred}{head_args}")
        context, result = head_args
        state = [x for x in body if x[0] == "state_before"]
        steps = [x for x in body if x[0].startswith("step_")]
        blockers = [x for x in body if x[0] in ("wall", "open")]
        if len(state) != 1 or len(steps) != 1 or len(blockers) != 1:
            raise PopperBackendError(f"rule does not match the four-literal transition template: {rule_text}")
        state_args = state[0][1]
        step_pred, step_args = steps[0]
        blocker_pred, blocker_args = blockers[0]
        if len(state_args) != 2 or len(step_args) != 3 or len(blocker_args) != 2:
            raise PopperBackendError(f"malformed transition rule: {rule_text}")
        if state_args[0] != context or step_args[0] != context:
            raise PopperBackendError(f"context variable is not shared safely: {rule_text}")
        if state_args[1] != step_args[2] or blocker_args[0] != context or blocker_args[1] != step_args[1]:
            raise PopperBackendError(f"rule does not preserve source/target bindings: {rule_text}")
        direction = step_pred.removeprefix("step_")
        if direction not in _DIRECTIONS:
            raise PopperBackendError(f"unsupported transition direction: {rule_text}")
        target = step_args[1]
        source = step_args[2]
        if blocker_pred == "wall" and result != source:
            raise PopperBackendError(f"wall rule must keep the source state: {rule_text}")
        if blocker_pred == "open" and result != target:
            raise PopperBackendError(f"open rule must return the adjacent target: {rule_text}")
        wall_literal = "wall({})".format(target)
        if blocker_pred == "open":
            wall_literal = "not wall({})".format(target)
        converted.append(
            "state_at({result},T+1):-time(T),adjacent({direction},{target},{source}),"
            "state_at({source},T),action({direction},T),{wall}.".format(
                result=result,
                direction=direction,
                target=target,
                source=source,
                wall=wall_literal,
            )
        )
    return "\n".join(converted) + "\n"


def learn_and_convert_transition_hypothesis(*args, **kwargs) -> Tuple[str, str]:
    """Convenience wrapper returning both contextual and planner hypotheses."""

    contextual = learn_transition_hypothesis(*args, **kwargs)
    return contextual, convert_popper_hypothesis_to_planner(contextual)
