import argparse
import json
import os
import re
from datetime import datetime
from glob import glob
from typing import List, Dict, Union

import datasets
import pandas as pd
from tqdm import tqdm

from super.agent.agent import AgentStep
from super.agent.utils import logger
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


def evaluate_checkpoints(gold_checkpoints: List[Dict], agent_history: List[Union[EnvironmentStep, Dict]]) -> float:
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
        logger.info(f"Checkpoint '{checkpoint}': {'Hit' if hit else 'Miss'}")
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
            logger.warning(f"Unsupported entrypoint file type: {entrypoint}")
            return None

    return 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-dir", "-t", required=True)
    parser.add_argument("--set", type=str, choices=["Expert", "Masked", "Auto"], required=True)
    return parser.parse_args()


def get_metrics(trajectory, task):
    metrics = {
        "submitted": 0,
        "output_match": None,
        "landmarks": None,
        "entrypoint": None,
    }

    last_action = trajectory[-1].action

    submission = None
    if last_action["type"] == "submit" and "content" in last_action:
        if last_action["content"]:
            metrics["submitted"] = 1
            submission = last_action["content"]

    task_name = task["task_id"]
    print(f"Task {task_name}\n****agent submission: {submission}**")

    gold_answer = json.loads(task["answer"]) if task.get("answer") else None
    if gold_answer:
        print(f"Task {task_name} gold answer: {gold_answer}")
        metrics["output_match"] = evaluate(gold_answer, submission)

        print(f"Task {task_name} output match metric: {metrics['output_match']}")

    gold_landmarks = task.get("landmarks")
    if gold_landmarks:
        metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    gold_entrypoint = task.get("entrypoint")
    if gold_entrypoint:
        metrics["entrypoint"] = evaluate_entrypoint(gold_entrypoint, trajectory)

    return metrics


def run_evaluation():
    args = parse_args()

    tasks = datasets.load_dataset("allenai/super", split=args.set)
    tasks = {task["task_id"]: task for task in tasks}

    tasks_found_in_trajectory_path = glob(os.path.join(args.trajectories_dir, "*"))
    if not tasks_found_in_trajectory_path:
        raise ValueError(
            f"No tasks found in the trajectory path: {args.trajectories_dir} ({os.path.abspath(args.trajectories_dir)})")

    all_metrics = []
    for task_dir in tqdm(tasks_found_in_trajectory_path):
        if not os.path.exists(os.path.join(task_dir, "metadata.json")):
            continue
        metadata = json.load(open(os.path.join(task_dir, "metadata.json")))
        task_name = metadata["task"]["task_id"]
        task = tasks.get(task_name)

        if not task:
            logger.warning(f"Task {task_name} not found in the tasks file")
            continue

        logger.info(f"*** Task {task_name} ***")

        if not os.path.exists(os.path.join(task_dir, "history.json")):
            continue

        with open(os.path.join(task_dir, "history.json")) as f:
            trajectory = json.load(f)
            trajectory = [AgentStep(**step) for step in trajectory["history"]]

        all_metrics.append(get_metrics(trajectory, task))

    df = pd.DataFrame(all_metrics)
    print(df.mean())



if __name__ == "__main__":
    run_evaluation()