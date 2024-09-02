import asyncio
import logging
import socket

import requests
from fastapi import FastAPI, HTTPException
from typing import Dict
from hypercorn.asyncio import serve
from hypercorn.config import Config

ARGS_REQUIRED = 4
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("API")


def create_app(network):
    app = FastAPI()
    app.shutdown = asyncio.Event()
    app.stats = {'inlined': 0, 'not_inlined': 0}

    @app.post(f"/predict")
    async def predict(data: Dict[str, str]):
        if len(data) != ARGS_REQUIRED:
            raise HTTPException(500, f"Please provide exactly {ARGS_REQUIRED} arguments.")

        decision = network.activate([int(d) for d in data.values()])[0] >= 0.5

        if decision:
            app.stats['inlined'] += 1
        else:
            app.stats['not_inlined'] += 1
        return {"result": decision}

    @app.post(f"/test")
    async def test():
        return {"result": True}

    @app.post(f"/shutdown")
    async def shutdown():
        LOGGER.info("Inlining stats: %s", app.stats)
        app.shutdown.set()

    return app


def find_available_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def deploy_endpoint(app):
    global LOGGER
    #port = find_available_port()
    port = 8001
    LOGGER.info(f"Launching API endpoint on port {port}")
    config = Config()
    config.bind = ["0.0.0.0:" + str(port)]
    config.accesslog = "-"
    config.loglevel = "CRITICAL"  # turn off logging of individual requests
    return await serve(app, config, shutdown_trigger=app.shutdown.wait)


def remove_endpoint():
    global LOGGER
    requests.post("http://0.0.0.0:8001/shutdown")
    LOGGER.info("Endpoint has been successfully closed.")


class MockNetwork:
    def activate(self, _):
        return list([1.0])  # True


if __name__ == "__main__":
    LOGGER.info("Starting API with mock network.")
    app = create_app(MockNetwork())
    task = deploy_endpoint(app)
    LOGGER.info("Endpoint with mock network has been deployed.")

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        remove_endpoint()
