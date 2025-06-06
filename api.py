import asyncio
import time
from distutils.util import strtobool
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from hypercorn.asyncio import serve
from hypercorn.config import Config
import logging
import numpy as np
import os
import requests
import socket
from typing import Dict, Any


load_dotenv(override=True)
USE_GRAPHS = bool(strtobool(os.getenv('USE_GRAPHS', 'False')))
if USE_GRAPHS:
    PARAMETERS_REQUIRED = 178
else:
    PARAMETERS_REQUIRED = 4
NODE_CLASS_ENCODINGS = np.identity(PARAMETERS_REQUIRED)
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("API")


def create_app(network, use_graphs):
    app = FastAPI()
    app.shutdown = asyncio.Event()
    app.stats = {'inlined': 0, 'not_inlined': 0}
    LOGGER.info("Creating endpoint in " + ("graph input mode" if use_graphs else "legacy baseline mode"))

    @app.post(f"/predict")
    async def predict(data: Dict[str, Any]):
        # start = time.perf_counter_ns()
        if use_graphs:
            if len(data) != 2 or not data["nodes"] or not data["edges"]:
                raise HTTPException(500, f"Payload should contain \"nodes\" and \"edges\" keys.")
            # conversion_start = time.perf_counter_ns()
            nodes_in = np.array([NODE_CLASS_ENCODINGS[int(n["nodeType"])] for n in data["nodes"]])
            edges_in = np.array(data["edges"]).T
            # LOGGER.info(f"json to np array conversion: {(time.perf_counter_ns()-conversion_start)/1000000} ms")
            try:
                decision = network.activate(nodes_in, edges_in)[0] >= 0.5
            except Exception as e:
                LOGGER.warning(e)
                raise e
            # LOGGER.info(f"Model inference took: {(time.perf_counter_ns()-start)/1000000} ms for {len(data['nodes'])} nodes")
        else:
            if len(data) != PARAMETERS_REQUIRED:
                raise HTTPException(500, f"Please provide exactly {PARAMETERS_REQUIRED} parameters.")

            try:
                decision = network.activate([int(d) for d in data.values()])[0] >= 0.5
            except Exception as e:
                LOGGER.warning(e)
                raise e
            # LOGGER.info(f"Model inference took: {(time.perf_counter_ns()-start)/1000000} ms")

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


async def deploy_endpoint(app, port):
    global LOGGER
    LOGGER.info(f"Launching API endpoint on port {port}")
    config = Config()
    config.bind = ["0.0.0.0:" + str(port)]
    config.accesslog = "-"
    config.loglevel = "CRITICAL"  # turn off logging of individual requests
    return await serve(app, config, shutdown_trigger=app.shutdown.wait)


def remove_endpoint(port):
    global LOGGER
    LOGGER.info("Shutting down endpoint..")
    requests.post(f"http://0.0.0.0:{port}/shutdown")
    LOGGER.info("Endpoint has been successfully closed.")


class MockNetwork:
    def activate(self, *_):
        return list([1.0])  # True


if __name__ == "__main__":
    LOGGER.info("Starting API with mock network.")
    port = 8001
    app = create_app(MockNetwork(), USE_GRAPHS)
    task = deploy_endpoint(app, port)
    LOGGER.info("Endpoint with mock network has been deployed.")

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        remove_endpoint(port)
