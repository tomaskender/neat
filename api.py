import asyncio
from distutils.util import strtobool
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
import os
import requests
import socket
from typing import Dict, Any


load_dotenv(override=True)
USE_GRAPHS = bool(strtobool(os.getenv('USE_GRAPHS', 'False')))
PARAMETERS_REQUIRED = 4 # should be the same as input parameters in NEAT config
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("API")


def create_app(network, use_graphs):
    app = FastAPI()
    app.shutdown = asyncio.Event()
    app.stats = {'inlined': 0, 'not_inlined': 0}

    @app.post(f"/predict")
    async def predict(data: Dict[str, Any]):
        if use_graphs:
            if len(data) != 2 or not data["nodes"] or not data["edges"]:
                raise HTTPException(500, f"Payload should contain \"nodes\" and \"edges\" keys.")

            decision = network.activate(data["nodes"], data["edges"])[0] >= 0.5
        else:
            if len(data) != PARAMETERS_REQUIRED:
                raise HTTPException(500, f"Please provide exactly {PARAMETERS_REQUIRED} parameters.")

            decision = network.activate([int(d) for d in data.values()])[0] >= 0.5

        if decision:
            app.stats['inlined'] += 1
        else:
            app.stats['not_inlined'] += 1
        return {"result": decision}

    @app.get(f"/test")
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
    app = create_app(MockNetwork(), USE_GRAPHS)
    task = deploy_endpoint(app)
    LOGGER.info("Endpoint with mock network has been deployed.")

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        remove_endpoint()
