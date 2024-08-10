from abc import ABC, abstractmethod
from typing import Dict, Tuple

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from super.env import Environment


class ACI(ABC):
    """
    A wrapper/ACI (agent-computer interface) for environments.
    """
    def __init__(self, env: "Environment", **kwargs):
        self._env = env

    @abstractmethod
    def step(self, action: Dict[str, str]) -> Tuple[str, bool]:
        pass

