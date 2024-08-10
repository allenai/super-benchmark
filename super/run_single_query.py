import os
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

from super.agent import Agent
from super.env import Environment
from super.env.edit_aci import EditACI

DEFAULT_MODEL_ENGINE = "gpt-4o-2024-08-06"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def add_experiment_args(parser):
    parser.add_argument("--model-engine", type=str, default=DEFAULT_MODEL_ENGINE)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-backend", type=str, help="The environment to use",
                        choices=["local", "jupyter", "modal"],
                        default="local")
    parser.add_argument("--agent", type=str, help="The agent to run",
                        choices=["react-super"],
                        default="react-super")
    return parser


def run_single_query():
    parser = ArgumentParser()
    add_experiment_args(parser)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    task = args.query

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output_dir) / run_id
    ctxt = nullcontext()
    if args.env_backend == "modal":
        from super.env.start_modal_app import stub
        ctxt = stub.run()
    with ctxt:
        if args.agent == "react-super":
            env = Environment.load_env(backend=args.env_backend, logs_output_dir=output_path)
            env.register_action_type("edit", EditACI(env))
            agent = Agent(task, args.model_engine, max_iterations=args.max_iterations, logs_output_dir=output_path)

            for i in range(args.max_iterations):
                action = agent.next_action()
                step = env.step(action)
                agent.step_completed(step)

                if step.done:
                    break
        else:
            raise ValueError(f"Invalid agent: {args.agent}")

    last_action = env.get_history()[-1].action
    print(f"Agent submission/last action: {last_action}")


if __name__ == "__main__":
    run_single_query()
