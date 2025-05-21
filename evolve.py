import asyncio
from distutils.util import strtobool
from dotenv import load_dotenv
import json
import logging
import neat
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
import sys
import subprocess
import threading
import time

from urllib3 import Retry

from api import deploy_endpoint, create_app, remove_endpoint
from gnn.graph_network import GraphNetwork

load_dotenv(override=True)
USE_GRAPHS = bool(strtobool(os.getenv('USE_GRAPHS', 'False')))
HOME_DIR = Path(os.getenv("GRAAL_REPO_DIR")) / "vm"
BENCH_RESULTS = Path(HOME_DIR) / "bench-results.json"
# BENCHMARKS = ["akka-uct", "db-shootout", "dotty", "finagle-chirper", "finagle-http", "fj-kmeans", "future-genetic", "mnemonics", "par-mnemonics", "philosophers", "reactors", "rx-scrabble", "scala-doku", "scala-kmeans", "scala-stm-bench7", "scrabble"]
BENCHMARKS = ["akka-uct", "db-shootout", "dotty"]
BENCHMARK_METRICS = ["time", "reachable-methods", "binary-size", "max-rss"]

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("NEAT evolution")


def get_benchmark_cmd(benchmark):
    return "mx --java-home={} --env ni-ce benchmark \"renaissance-native-image:{}\"  --  --jvm=native-image --jvm-config=default-ce -Dnative-image.benchmark.extra-image-build-argument=--parallelism=12".format(os.getenv("JAVA_HOME"), benchmark)


async def build_network_and_deploy(genome, config, port, is_graph_mode):
    if is_graph_mode:
        net = GraphNetwork.create(genome, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

    app = create_app(net, is_graph_mode)
    await deploy_endpoint(app, port)

def eval_genome(genome, config):
    global cache
    if genome.fitness is not None:
        LOGGER.info(f"Found cached genome {genome.key} with fitness {genome.fitness}. Skipping re-evaluation.")
        return genome.fitness

    LOGGER.info(f"Scheduling network endpoint deployment")
    port = 8001
    loop = asyncio.new_event_loop()
    task = loop.create_task(build_network_and_deploy(genome, config, port, USE_GRAPHS))
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    time.sleep(2)

    LOGGER.info(f"Waiting for endpoint to come online..")

    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=2,
                    status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    
    if s.get(f"http://0.0.0.0:{port}/test").status_code != 200:
        LOGGER.info("Could not verify endpoint with a test request")
        exit(1)
    LOGGER.info("Verified connectivity to endpoint")

    start = time.perf_counter()
    fitness = 0.0

    for benchmark in BENCHMARKS:
        # run benchmark, graal inliner uses endpoint running on `server.config.port`
        LOGGER.info(f"Launching benchmark {benchmark}")

        # delete previous benchmark log
        BENCH_RESULTS.unlink(missing_ok=True)

        completed_process = subprocess.run(
            f"cd {HOME_DIR} && {get_benchmark_cmd(benchmark)}",
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
        LOGGER.info(f"Benchmarking subprocess completed with {completed_process.returncode}")

        stats = {}
        if completed_process.returncode == 0:
            with open(BENCH_RESULTS) as json_file:
                data = json.load(json_file)

                for q in data["queries"]:
                    cols = ["metric.name", "metric.object"]
                    for col in cols:
                        if col in q and q[col] in BENCHMARK_METRICS:
                            stats[q[col]] = q["metric.value"]
            logging.debug(f"{genome} produced following benchmark stats for {benchmark}: {stats}")
        else:
            LOGGER.info(f"Benchmarking subprocess completed with error:\n{completed_process.stdout}")
            stats = dict.fromkeys(BENCHMARK_METRICS, 999_999_999)

        fitness += stats["reachable-methods"]
    LOGGER.info(f"Benchmarking took {time.perf_counter()-start}s")

    # remove endpoint after benchmarking is done
    remove_endpoint(port)
    task.cancel()
    time.sleep(2)

    # workaround due to NEAT ignoring activation setting in config: always perform fitness maximization for negative fitness
    genome.fitness = -fitness
    return -fitness


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    LOGGER.info("Initial population has been created.")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ThreadedEvaluator(1, eval_genome)

    LOGGER.info("Running simulation.")
    winner = p.run(pe.evaluate, 5)

    LOGGER.info(f"Best genome: {winner}")


if __name__ == '__main__':
    if USE_GRAPHS:
        config_path = Path(__file__).parent / 'config-gnn'
    else:
        config_path = Path(__file__).parent / 'config-feedforward'
    run(config_path)
