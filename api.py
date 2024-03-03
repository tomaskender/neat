import logging
import socket
import time

from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn


ARGS_REQUIRED = 4
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("API")


def create_app(network):
    app = FastAPI()

    @app.post(f"/predict")
    async def predict(args: List[int]):
        if len(args) != ARGS_REQUIRED:
            raise HTTPException(500, f"Please provide exactly {ARGS_REQUIRED} arguments.")
        return {"result": network.activate(args)}

    return app


def find_available_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch_service(network):
    global LOGGER
    #port = find_available_port()
    port = 8001
    LOGGER.info(f"Launching API endpoint on port {port}")
    app = create_app(network)
    uvicorn.run(app=app, port=port)
    return port


class MockNetwork:
    def activate(self, _):
        return True


if __name__ == "__main__":
    LOGGER.info("Starting API with mock network.")
    launch_service(MockNetwork())

    while 1:
        time.sleep(1)
