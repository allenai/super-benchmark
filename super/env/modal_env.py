import requests
import secrets
import time
from time import sleep
from urllib.parse import urlparse

from super.utils import logger
from super.env.jupyter_env import JupyterEnv
from super.env.start_modal_app import run_jupyter_in_modal, get_modal_url_queue

# Function to check if Jupyter is running
def is_jupyter_running(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        pass
    return False


class ModalEnv(JupyterEnv):

    def __init__(self, rank: int = -1, **kwargs):
        """
        :param reset_container_if_exists if True and the container already exists, it will be reset to its "fresh" state
        """
        self.jupyter_token = secrets.token_hex(8)
        # make sure to use a unique queue rank, if not passed
        if rank == -1:
            self.rank = (time.time() % 100) + 1
        else:
            self.rank = rank
        try:
            hostname, jupyter_job = self.init_modal_env()
            super().__init__(host=hostname, server_port=None, token=self.jupyter_token, use_secure=True, **kwargs)
            self._working_dir = "/content"
            self.run_command(f"mkdir -p {self._working_dir}")
            self.run_command(f"%cd {self._working_dir}")
        except Exception as e:
            if jupyter_job:
                jupyter_job.cancel()
            raise e

    def init_modal_env(self):
        # Always init a new modal env (has a built in timeout) for simplicity
        jupyter_job = run_jupyter_in_modal.spawn(self.jupyter_token, self.rank)
        # Get the modal host from the queue
        modal_url_queue = get_modal_url_queue(self.rank)
        try:
            self.modal_host = modal_url_queue.get(timeout=180)
        except Exception as e:
            logger.error("Unable to get the modal host from the queue. Exiting...")
            raise e
        # Wait for the server to start. If it doesn't start in 3 minutes, raise an exception
        max_sleep = 180
        running = False
        while max_sleep > 0:
            running = is_jupyter_running(self.modal_host)
            if running is False:
                sleep(2)
                max_sleep -= 2
            else:
                break
        if running is None:
            raise Exception("Failed to start server or started incorrectly!")
        parsed_url = urlparse(self.modal_host)
        host_name = parsed_url.netloc + parsed_url.path
        return host_name, jupyter_job