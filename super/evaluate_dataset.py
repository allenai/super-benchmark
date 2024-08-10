import os
import re
from datetime import datetime
from typing import List, Dict, Union

from super.env.environment import EnvironmentStep


def evaluate(predicted, gold, float_epsilon: float = 1e-2) -> float:
    if type(gold) == int:
        gold = float(gold)
    if type(predicted) == int:
        predicted = float(predicted)

    if type(gold) != type(predicted):
        return 0.0

    if type(gold) == list:
        if len(gold) == 0:
            raise ValueError("Gold is empty")
        return sum([evaluate(p, g) for p, g in zip(predicted, gold)]) / len(gold)

    if type(gold) == dict:
        if len(gold) == 0:
            raise ValueError("Gold is empty")
        return sum([evaluate(gv, predicted.get(gk, None), float_epsilon=float_epsilon) for gk, gv in gold.items()]) / len(gold)

    if type(gold) == str:
        return float(predicted.strip() == gold.strip())

    if type(gold) == float:
        return float(abs(predicted - gold) < float_epsilon)

    raise NotImplementedError


def evaluate_checkpoints(gold_checkpoints: List[Dict], agent_history: List[Union[EnvironmentStep, Dict]], verbose: bool = False) -> float:
    """
    Evaluate if the agent has gone through some gold checkpoints by looking for certain outputs in the agent's history,
    e.g. "Training completed..."
    """
    checkpoints_hit = []
    if len(agent_history) and type(agent_history[0]) == dict:
        agent_history = [EnvironmentStep(**step) for step in agent_history]

    for checkpoint in gold_checkpoints:
        hit = False
        always_executed = False
        for step in agent_history:
            if re.search(checkpoint, step.observation.replace("\n", " ")):
                hit = True
                break
        if always_executed:
            continue
        checkpoints_hit.append(hit)
        if verbose:
            print(f"Checkpoint {checkpoint['check_type']} '{checkpoint['contents']}': {'Hit' if hit else 'Miss'}")
    return sum(checkpoints_hit) / len(checkpoints_hit)


def evaluate_entrypoint(entrypoint: str, agent_history: List[Union[EnvironmentStep, Dict]]) -> float:
    """
    Evaluate if a specific entrypoint (i.e. a Python or bash script) was successfully executed by the agent.
    We check if:
    * Entrypoint was run for more than 10 seconds and ended
    * No exceptions were raised
    """
    entrypoint_filename = os.path.basename(entrypoint)

    for step in agent_history:
        if entrypoint.endswith(".py") or entrypoint.endswith(".sh") or entrypoint.endswith(".bash"):
            command = None
            if "content" in step.action:
                command = step.action["content"]
            elif "execute_action" in step.action:
                # (swe-agent)
                command = step.action["execute_action"]
            if not command:
                continue
            ran_entrypoint = entrypoint_filename in command
            if not ran_entrypoint:
                # check module name
                ran_entrypoint = entrypoint_filename.replace(".py", "").replace("/", ".") in command and "python -m" in command

            if not ran_entrypoint:
                continue

            if "Traceback" in step.observation:
                continue

            if f"{entrypoint_filename}: error" in step.observation:
                # this is the error format of wrong arguments usage
                continue

            execution_start_time = datetime.strptime(step.execution_start_time, "%Y-%m-%d %H:%M:%S.%f") if type(step.execution_start_time) == str else step.execution_start_time
            execution_end_time = datetime.strptime(step.execution_end_time, "%Y-%m-%d %H:%M:%S.%f") if type(step.execution_end_time) == str else step.execution_end_time
            step_execution_time = execution_end_time - execution_start_time
            if step_execution_time.total_seconds() < 10:
                continue

            return 1.0
        else:
            print("Skipping")
            return None

    return 0.0
