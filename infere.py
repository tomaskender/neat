import argparse
import asyncio
from distutils.util import strtobool
import json
import logging
from pathlib import Path
import threading
import time
import neat
import neat.genome
import requests
from urllib3 import Retry

from api import remove_endpoint
from evolve import build_network_and_deploy

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("INFERE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Infere',
                    description='Deploy inlining model to API for inference')
    parser.add_argument('filename', help='Path to JSON containing config of inlining model')
    args = parser.parse_args()

    genome_cfg = json.load(open(args.filename))
    genome = neat.DefaultGenome(1)
    port = 8001

    is_graph_mode = bool(genome_cfg["metadata"]["graph_mode"]) is True
    LOGGER.info("Starting API with pre-trained network in {} mode.".format("graph" if is_graph_mode else "legacy"))

    if genome_cfg["metadata"]["graph_mode"] is True:
        config_file = Path(__file__).parent / 'config-gnn'
    else:
        config_file = Path(__file__).parent / 'config-feedforward'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    genome.nodes = {}
    for node in genome_cfg["nodes"]:
        node_key = node["key"]
        gene = neat.genome.DefaultNodeGene(node_key)
        gene.bias = node["bias"]
        gene.response = node["response"]
        gene.activation = node["activation"]
        gene.aggregation = node["aggregation"]
        genome.nodes[node_key] = gene

    genome.connections = {}
    for node in genome_cfg["connections"]:
        [idx1, idx2] = node["key"]
        conn_key = (idx1, idx2)
        conn = neat.genome.DefaultConnectionGene(conn_key)
        conn.enabled = node["enabled"]
        conn.weight = node["weight"]
        genome.connections[conn_key] = conn

    loop = asyncio.new_event_loop()
    task = loop.create_task(build_network_and_deploy(genome, config, port, is_graph_mode))
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    time.sleep(2)

    LOGGER.info(f"Waiting for endpoint to come online..")

    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=2,
                    status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    
    if s.get(f"http://0.0.0.0:{port}/test").status_code != 200:
        LOGGER.info("Could not verify endpoint with a test request")
        exit(1)
    LOGGER.info("Verified connectivity to endpoint")


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        remove_endpoint(port)
