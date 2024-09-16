import datetime
import json
import uuid
from typing import Tuple, Dict

import requests
from websocket import create_connection, WebSocketTimeoutException

from super.utils import logger
from super.env import Environment
from super.env.utils import timeout_call


def prepare_execute_message(code):
    msg_type = "execute_request"
    content = {"code": code, "silent": False}
    hdr = {
        "msg_id": uuid.uuid1().hex,
        "username": "test",
        "session": uuid.uuid1().hex,
        "data": datetime.datetime.now().isoformat(),
        "msg_type": msg_type,
        "version": "5.0",
    }
    msg = {"header": hdr, "parent_header": hdr, "metadata": {}, "content": content}
    return msg


class JupyterEnv(Environment):
    def __init__(self, host: str = "localhost", server_port: int = 8889, token: str = "password",
                 use_secure = False, timeout_per_command=60*5, **kwargs):
        """
        :param reset_container_if_exists if True and the container already exists, it will be reset to its "fresh" state
        """
        super().__init__(**kwargs)
        self.server_port = server_port
        self.token = token
        self._headers = {"Authorization": f"Token {self.token}"}
        self.use_secure = use_secure
        self.host = host
        self._timeout_per_command = timeout_per_command

        self._intermediate_output = []
        self._msgs_history = []  # helpful for debugging websockets messages
        self.ws = None
        self.timed_out_unanswered = False

        self.timeout_message = f"""\n\nYour command has already run for {self._timeout_per_command} seconds. It is still running. You can continue to wait or interrupt it with Thought: ... \nAction: interrupt\n```\n``` \nor:\nThought: ... \nAction: continue_wait\n```\n```"""

        self.init_container_notebook()

    def _step(self, action: Dict[str, str]) -> Tuple[str, bool]:
        done = False
        if action["type"] == "execute":
            observation = self.run_command(action["content"])
        elif action["type"] == "submit":
            observation = ""
            done = True
        elif action["type"] == "interrupt":
            self.interrupt_kernel()
            observation = "Kernel interrupted."
        elif action["type"] == "continue_wait":
            observation = self.continue_after_timeout()
        else:
            observation = f"Invalid action type: {action['type']}"
        return observation, done

    def init_container_notebook(self):
        # Create a kernel
        self._jupyter_server = f"{self.host}:{self.server_port}" if self.server_port else self.host
        protocol = "https" if self.use_secure else "http"
        url = f"{protocol}://{self._jupyter_server}/api/kernels"
        logger.info(f"Creating a new kernel at {url}")

        response = requests.post(url, headers=self._headers)
        if "id" not in json.loads(response.text):
            raise Exception(f"Failed to create a kernel! No id in {response.text}. Check host, port and token.")
        else:
            self.kernel_id = json.loads(response.text)["id"]

        self._create_connection()

    def interrupt_kernel(self):
        self._jupyter_server = f"{self.host}:{self.server_port}" if self.server_port else self.host
        protocol = "https" if self.use_secure else "http"
        url = f"{protocol}://{self._jupyter_server}/api/kernels/{self.kernel_id}/interrupt"
        with requests.post(url, headers=self._headers) as response:
            assert response.status_code == 204, f"Failed to interrupt the kernel: {response}"
        self._create_connection()
        self.timed_out_unanswered = False
        self.run_command("pass")

    def _create_connection(self):
        if self.ws:
            self.ws.close()  # Close existing connection if any before creating a new one
        protocol = "wss" if self.use_secure else "ws"
        self.ws = create_connection(
            f"{protocol}://{self._jupyter_server}/api/kernels/{self.kernel_id}/channels",
            header=self._headers
        )

    def continue_after_timeout(self):
        return self.run_command(continue_after_timeout=True)

    def run_command(self, command: str = None, continue_after_timeout: bool = False) -> str:
        if continue_after_timeout:
            self.timed_out_unanswered = False
            self._intermediate_output = []
        else:
            if self.timed_out_unanswered:
                # this happens if the previous command was timed out, and the agent did not call continue_after_timeout or interrupt_kernel.
                # in this case, we should force an interrupt.
                self.timed_out_unanswered = False
                self.interrupt_kernel()
            self._initiate_ws(command)

        try:
            return timeout_call(self.retrieve_output, self._timeout_per_command)
        except TimeoutError as e:
            self.timed_out()
            return "".join(self._intermediate_output)

    def timed_out(self):
        self.timed_out_unanswered = True
        self._intermediate_output.append(self.timeout_message)

    def _initiate_ws(self, command: str):
        try:
            self.ws.send(json.dumps(prepare_execute_message(command)))
        except BrokenPipeError:
            self._create_connection()
            self.ws.send(json.dumps(prepare_execute_message(command)))

        self._intermediate_output = []

    def retrieve_output(self) -> str:
        status = "success"
        execute_reply_msg_id = None
        self.ws.settimeout(60*60*24)

        while True:
            try:
                recv_obj = self.ws.recv()
            except WebSocketTimeoutException:
                self.timed_out()
                status = "timeout"
                break
            except BrokenPipeError as e:
                logger.error("Broken pipe error. Message history:")
                for msg in self._msgs_history:
                    logger.info(msg)
                raise e
            if not recv_obj:
                continue
            rsp = json.loads(recv_obj)
            self._msgs_history.append(rsp)
            msg_type = rsp["msg_type"]
            if msg_type == "error":
                self._intermediate_output.append(f"Error/Traceback: {rsp['content']['ename']}: {rsp['content']['evalue']}")
                for tb in rsp["content"]["traceback"]:
                    self._intermediate_output.append(tb)
                status = "error"
                break
            elif msg_type == "stream":
                self._intermediate_output.append(rsp["content"]["text"])
                logger.info(rsp["content"]["text"])
            elif msg_type == "execute_result":
                self._intermediate_output.append(rsp["content"]["data"]["text/plain"])

            current_msg_id = self._get_msg_id_from_rsp(rsp)
            if execute_reply_msg_id and current_msg_id == execute_reply_msg_id - 1:
                break
            if (
                    msg_type == "execute_reply"
                    and rsp["metadata"].get("status") == "ok"
                    and rsp["metadata"].get("dependencies_met", False)
            ):
                # check if we didn't skip any stream messages - unfortunately, sometimes stream messages come after the execute_reply message. This is not the most robust way to handle this, but it seems stable enough.
                # we check the message id of the execute_reply (finished message) and the message id of the last stream message. If the stream message id is higher, we continue receiving messages until we get to the message id of execute_reply minus 1.
                try:
                    execute_reply_msg_id = self._get_msg_id_from_rsp(rsp)
                    last_message_id = self._get_msg_id_from_rsp(self._msgs_history[-2])
                    if execute_reply_msg_id - last_message_id > 1:
                        # timeout can now be short since last message was already received
                        logger.info("*** execute_reply was received before the last stream message. Continuing to receive messages until the last stream message is received. ***")
                        self.ws.settimeout(5)
                        continue
                except Exception as e:
                    break
                break
        return "".join(self._intermediate_output)

    @staticmethod
    def _get_msg_id_from_rsp(rsp):
        return int(rsp["header"]["msg_id"].split("_")[-1])