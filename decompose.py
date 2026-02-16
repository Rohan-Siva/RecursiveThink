import json
from typing import List

from subgoal import SubgoalState
from branching_prompts import build_decompose_prompt
from parser import extract_json


def decompose_problem(model, problem: str) -> List[SubgoalState]:
    system_prompt, user_prompt = build_decompose_prompt(problem)

    response = model.generate(system_prompt, user_prompt, json_mode=True)
    raw = response.text

    json_str = extract_json(raw)
    if not json_str:
        return _fallback_single_subgoal(problem)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return _fallback_single_subgoal(problem)

    subgoals_raw = data.get("subgoals", [])
    if not subgoals_raw or not isinstance(subgoals_raw, list):
        return _fallback_single_subgoal(problem)

    subgoals = []
    for sg in subgoals_raw:
        sg_id = int(sg.get("id", len(subgoals) + 1))
        desc = str(sg.get("description", ""))
        deps = sg.get("dependencies", [])
        if not isinstance(deps, list):
            deps = []
        deps = [int(d) for d in deps]
        subgoals.append(SubgoalState(
            subgoal_id=sg_id,
            description=desc,
            dependencies=deps,
        ))

    if not subgoals:
        return _fallback_single_subgoal(problem)

    return subgoals


def _fallback_single_subgoal(problem: str) -> List[SubgoalState]:
    return [SubgoalState(
        subgoal_id=1,
        description=f"Solve the problem: {problem}",
        dependencies=[],
    )]


def topological_order(subgoals: List[SubgoalState]) -> List[SubgoalState]:
    by_id = {sg.subgoal_id: sg for sg in subgoals}
    visited = set()
    order = []

    def visit(sg_id):
        if sg_id in visited:
            return
        visited.add(sg_id)
        sg = by_id.get(sg_id)
        if sg is None:
            return
        for dep in sg.dependencies:
            visit(dep)
        order.append(sg)

    for sg in subgoals:
        visit(sg.subgoal_id)

    return order
