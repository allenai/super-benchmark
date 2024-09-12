import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import datasets

from super.env.edit_aci import EditACI
from super.agent import Agent
from super.env import Environment
from super.evaluate_dataset import get_metrics
from super.run_single_query import add_experiment_args


def args_dataset_arguments(parser):
    parser.add_argument("--set", type=str, choices=["Expert", "Masked", "Auto"], required=True)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--agent-name", type=str, help="The agent to run", choices=["react-super"], default="react")
    parser.add_argument("--tasks-ids", type=str, nargs="+", help="The task ids to run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    return parser


def add_to_metadata_file(dir_path, metadata):
    file_path = os.path.join(dir_path, "metadata.json")
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        with open(file_path, "r") as f:
            data = json.load(f)
        data.update(metadata)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


def should_rerun_experiment(output_path):
    if os.path.exists(output_path / "stdout.txt"):
        log_contents = open(os.path.join(output_path / "stdout.txt")).read()
        if "Stopping app - uncaught exception raised locally" in log_contents:
            return True
        if "Stopping app - local client disconnected" in log_contents:
            return True
        if "✓ App aborted. View run at " in log_contents:
            return True
        if "Stopping app - keyboard interrupt received" in log_contents:
            return True
        if "Timed out waiting for logs. View logs at " in log_contents:
            return True
        if not log_contents.strip():
            return True
        return False
    else:
        return True


def run_single_task(task, args, run_id, rank):
    task_name = task["task_id"]
    if args.seed is not None:
        task_name += str(args.seed)
    output_path = Path(args.output_dir) / "runs" / run_id / task_name

    if os.path.exists(output_path) and not should_rerun_experiment(output_path):
        print(f"Task {task_name} already exists, skipping ({output_path})")
        return
    print(f"Starting task {task_name}")

    os.makedirs(output_path, exist_ok=True)

    add_to_metadata_file(output_path, {
        "run_command": " ".join(sys.argv),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "args": vars(args)
    })

    ctxt = nullcontext()

    if args.print_output:
        stdout_context = contextlib.nullcontext()
        stderr_context = contextlib.nullcontext()
    else:
        stdout_file_path = output_path / "stdout.txt"
        stdout_file = open(stdout_file_path, "w")
        stdout_context = contextlib.redirect_stdout(stdout_file)
        stderr_context = contextlib.redirect_stderr(stdout_file)

    if args.set == "Masked":
        max_token_limit = int(200_000 * args.budget_multiplier)  # $0.5 for gpt-4o-2024-08-06
    else:
        max_token_limit = int(300_000 * args.budget_multiplier)  # $0.75 for gpt-4o-2024-08-06

    with stdout_context, stderr_context:
        if args.env_backend == "modal":
            from super.env.start_modal_app import stub
            ctxt = stub.run()
        with ctxt:
            env_kwargs = {}
            if args.env_backend == "modal":
                add_to_metadata_file(output_path, {"modal_app_id": stub.app_id})
                env_kwargs["rank"] = rank
            if args.agent_name == "react":
                env = Environment.load_env(backend=args.env_backend, logs_output_dir=output_path, **env_kwargs)
                env.register_action_type("edit", EditACI(env))
                agent = Agent(
                    task["query"], args.model_engine,
                    max_iterations=args.max_iterations,
                    logs_output_dir=output_path,
                    max_compute_time=60 * 30,
                    max_context_tokens=max_token_limit,
                    cache_id=args.seed
                )

                if args.set == "Masked":
                    for cell in task["pre_execute_cells"]:
                        step = env.step(json.loads(cell))
                        agent.step_completed(step)

                for i in range(args.max_iterations):
                    action = agent.next_action()
                    step = env.step(action)
                    agent.step_completed(step)

                    if step.done:
                        break
            else:
                raise ValueError

            metrics = get_metrics(env.get_history(), task)

            with open(output_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
    print(f"Task {task_name} done")


def run_dataset():
    parser = argparse.ArgumentParser()
    parser = add_experiment_args(parser)
    parser = args_dataset_arguments(parser)
    args = parser.parse_args()

    run_id = f"{args.run_name}_{args.model_engine.split('/')[-1]}_{args.agent_name}_{args.set}"

    tasks = datasets.load_dataset("allenai/super", split=args.set)

    if args.tasks_ids:
        tasks = [task for task in tasks if task["task_id"] in args.tasks_ids]

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Loading tasks: {', '.join([task['task_id'] for task in tasks])}")

    if args.n_procs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
            for i, task in enumerate(tasks):
                executor.submit(run_single_task, task, args, run_id, i)
    else:
        for i, task in enumerate(tasks):
            run_single_task(task, args, run_id, i)


if __name__ == "__main__":
    run_dataset()
