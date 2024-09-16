import os
import subprocess
import time
import modal

from super.agent.utils import logger

DEMO_MODE = False


def get_modal_url_queue(rank=0):
    """rank allows running multiple instances of modal concurrently by keeping a unique queue for each instance."""
    return modal.Queue.from_name(f"jupyter-url-queue-{rank}", create_if_missing=True)

apt_packages = ["build-essential", "gcc", "cmake", "ccache", "curl", "git", "wget", "unzip",
                "sudo", "libreadline-dev", "clang", "zlib1g", "zlib1g-dev", "pkg-config",
                "jupyter", "openjdk-11-jdk"]
commands = []

gpu = None

stub = modal.Stub(
    image=modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install(apt_packages)
    .pip_install_from_requirements(os.path.join(os.path.dirname(__file__), "colab_pip_requirements.txt"))
    .run_commands(commands)
)

# Pass keys to the Modal environment
if modal.is_local() and DEMO_MODE:
    if os.getenv("TOGETHER_API_KEY") is None:
        raise ValueError("TOGETHER_API_KEY environment variable is not set.")
    if os.getenv("HUGGING_FACE_HUB_TOKEN") is None:
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable is not set.")
    local_secret = modal.Secret.from_dict({
        "TOGETHER_KEY": os.environ["TOGETHER_API_KEY"], # Pass this in case agent overrides TOGETHER_API_KEY
        "TOGETHER_API_KEY": os.environ["TOGETHER_API_KEY"],
        "HF_TOKEN": os.environ["HUGGING_FACE_HUB_TOKEN"]
        })
else:
    local_secret = modal.Secret.from_dict({})

modal_timeout = 3600
# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.
@stub.function(concurrency_limit=1, timeout=modal_timeout, secrets=[local_secret], gpu=gpu)
def run_jupyter_in_modal(jupyter_token: str, rank=0):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "server",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                f"--IdentityProvider.token={jupyter_token}",
                "--ServerApp.password=",
                "--ServerApp.allow_origin='*'",
                "--ServerApp.allow_remote_access=True",
            ],
            env={**os.environ},
        )

        logger.info(f"Jupyter available at => {tunnel.url}")
        modal_url_queue = get_modal_url_queue(rank)
        # Clean the queue; just in case there are any leftover urls
        while modal_url_queue.len():
            url = modal_url_queue.get()
            logger.info("Found extra url {} in the queue: {}. Deleting!".format(url,
                                                                          modal_url_queue))
        modal_url_queue.put(tunnel.url)
        try:
            end_time = time.time() + modal_timeout
            while time.time() < end_time:
                time.sleep(5)
            logger.error(f"Reached end of {modal_timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            logger.info("Exiting...")
        finally:
            jupyter_process.kill()


@stub.function()
def write_to_file_in_modal(content, file_name):
    with open(file_name, "w") as f:
        f.write(content)
