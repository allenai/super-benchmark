from io import StringIO
import traceback
from typing import Dict, Tuple

from IPython.core.interactiveshell import InteractiveShell
import sys

from super.utils import logger
from super.env import Environment


class IPythonEnv(Environment):
    """
    Use IPython's InteractiveShell to execute Python code or Shell commands. Note that shell commands
    will change the state of your local environment, since it is not run inside a Docker container. Use with caution.
    """

    def __init__(self):
        super().__init__()

        logger.warning("%" * 20 + " WARNING " + "%" * 20 +
              "\nIPythonEnv is not a secure environment. Use with caution.")
        self.shell = InteractiveShell()

    def _step(self, action: Dict[str, str]) -> Tuple[str, bool]:
        done = False
        if action["type"] == "execute":
            observation = self.run_command(action["content"])
        elif action["type"] == "submit":
            observation = ""
            done = True
        else:
            observation = f"Invalid action type: {action['type']}"
        return observation, done

    def run_command(self, command) -> str:
        # Redirect stdout to capture the output
        sys.stdout = StringIO()
        status = "success"
        try:
            # Execute the code using IPython's InteractiveShell
            result = self.shell.run_cell(command)
            if result.success:
                # If execution is successful, capture the output
                output = sys.stdout.getvalue()
                log = output
            else:
                # If execution failed, capture the error message
                error_message = str(result.error_in_exec)
                log = f"Error during execution:\n{error_message}"
                status = "error"
        except Exception as e:
            # If an exception occurs during execution, capture the error message
            log = f"Error during execution:\n{traceback.format_exc()}"
            status = "error"
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        return log
