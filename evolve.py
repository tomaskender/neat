import asyncio
import sys
import threading
import time

from dotenv import load_dotenv
import json
import logging
import neat
import os
from pathlib import Path
import subprocess
import visualize

from api import deploy_endpoint, remove_endpoint, create_app

load_dotenv(override=True)
HOME_DIR = Path(os.getenv("GRAAL_REPO_DIR")) / "vm"
BENCH_RESULTS = Path(HOME_DIR) / "bench-results.json"
BENCHMARKS = ["akka-uct", "db-shootout", "dotty", "finagle-chirper", "finagle-http", "fj-kmeans", "future-genetic", "mnemonics", "par-mnemonics", "philosophers", "reactors", "rx-scrabble", "scala-doku", "scala-kmeans", "scala-stm-bench7", "scrabble"]
BENCHMARK_METRICS = ["time", "reachable-methods", "binary-size", "max-rss"]

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("NEAT evolution")


def get_benchmark_cmd(benchmark):
    return "mx --java-home={} --env ni-ce benchmark \"renaissance-native-image:{}\"  --  --jvm=native-image --jvm-config=default-ce -Dnative-image.benchmark.extra-image-build-argument=--parallelism=12".format(os.getenv("JAVA_HOME"), benchmark)


async def build_network_and_deploy(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    app = create_app(net)
    await deploy_endpoint(app)


def eval_genome(genome, config):
    loop = asyncio.new_event_loop()
    task = loop.create_task(build_network_and_deploy(genome, config))
    thread = threading.Thread(target=loop.run_forever)
    thread.start()

    # delete previous benchmark log
    BENCH_RESULTS.unlink(missing_ok=True)

    start = time.perf_counter()

    # run benchmark, graal inliner uses endpoint running on `server.config.port`
    LOGGER.info(f"Launching benchmark {BENCHMARKS[0]}")
    completed_process = subprocess.run(
        f"cd {HOME_DIR} && {get_benchmark_cmd(BENCHMARKS[0])}",
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    LOGGER.info(f"Benchmarking subprocess completed with {completed_process.returncode}")
    LOGGER.info(f"Benchmarking took {time.perf_counter()-start}s")

    # Remove endpoint after benchmarking is done
    remove_endpoint()
    task.cancel()

    stats = {}
    if completed_process.returncode == 0:
        with open(BENCH_RESULTS) as json_file:
            data = json.load(json_file)

            for q in data["queries"]:
                cols = ["metric.name", "metric.object"]
                for col in cols:
                    if col in q and q[col] in BENCHMARK_METRICS:
                        stats[q[col]] = q["metric.value"]
        logging.debug(f"{genome} produced following benchmark stats: {stats}")
    else:
        LOGGER.info(f"Benchmarking subprocess completed with error:\n{completed_process.stdout}")
        stats = dict.fromkeys(BENCHMARK_METRICS, 999_999_999)

    # TODO experiment with different subgroups of collected metrics as fitness
    fitness = stats["reachable-methods"]
    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    LOGGER.info("Initial population has been created.")

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ThreadedEvaluator(1, eval_genome)

    LOGGER.info("Running simulation.")
    winner = p.run(pe.evaluate, 5)

    LOGGER.info(f"Best genome: {winner}")

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'Inline'}
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config-feedforward'
    run(config_path)
