import dataclasses
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Callable

from super.utils import logger
from super.env.aci import ACI


@dataclasses.dataclass
class EnvironmentStep:
    action: dict
    observation: str
    done: bool
    execution_start_time: datetime.datetime = None
    execution_end_time: datetime.datetime = None


class Environment(ABC):
    def __init__(self, logs_output_dir: str = None):
        self._working_dir = "/content"
        self._logs_output_dir = logs_output_dir
        self._history: List[EnvironmentStep] = []

        self._action_types: Dict[str, ACI] = {}

    def get_history(self) -> List[EnvironmentStep]:
        return self._history

    def register_action_type(self, action_type: str, fn: Callable[[Dict[str, str]], Tuple[str, bool]]):
        self._action_types[action_type] = fn

    def step(self, action: Dict[str, str]) -> EnvironmentStep:
        execution_start_time = datetime.datetime.now()
        if action["type"] == "exception":
            return EnvironmentStep(action, "", True)
        elif action["type"] in self._action_types:
            observation, done = self._action_types[action["type"]].step(action)
        else:
            observation, done = self._step(action)
        execution_end_time = datetime.datetime.now()
        step = EnvironmentStep(
            action,
            observation,
            done,
            execution_start_time=execution_start_time,
            execution_end_time=execution_end_time
        )
        self._history.append(step)

        return step

    @abstractmethod
    def _step(self, action: Dict[str, str]) -> Tuple[str, bool]:
        pass

    @abstractmethod
    def run_command(self, command: str) -> str:
        pass

    @classmethod
    def load_env(cls, backend: str, **kwargs):
        if backend == "local":
            from super.env.local_env import IPythonEnv
            return IPythonEnv()
        elif backend == "jupyter":
            from super.env.jupyter_env import JupyterEnv
            return JupyterEnv()
        elif backend == "modal":
            from super.env.modal_env import ModalEnv
            tries = 0
            while tries < 3:
                try:
                    return ModalEnv(**kwargs)
                except Exception as e:
                    logger.error("Failed to init Modal env. Retrying!")
                    tries += 1
        else:
            raise ValueError(f"Invalid backend: {backend}")
