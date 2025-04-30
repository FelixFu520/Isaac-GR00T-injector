import asyncio
import logging
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 5555,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(str(self._metadata)))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs_new = {}
                import numpy as np
                obs_new["video.image"] = obs["observation/image"][np.newaxis, ...]
                obs_new["video.wrist_image"] = obs["observation/wrist_image"][np.newaxis, ...]
                obs_new["state.x"] = np.array([[obs["observation/state"][0]]])
                obs_new["state.y"] = np.array([[obs["observation/state"][1]]])
                obs_new["state.z"] = np.array([[obs["observation/state"][2]]])
                obs_new["state.roll"] = np.array([[obs["observation/state"][3]]])
                obs_new["state.pitch"] = np.array([[obs["observation/state"][4]]])
                obs_new["state.yaw"] = np.array([[obs["observation/state"][5]]])
                obs_new["state.gripper"] = np.array([obs["observation/state"][6:]])
                obs_new["annotation.human.task_description"] = obs["prompt"]
                action = self._policy.get_action(obs_new)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
