from dotenv import load_dotenv
import json
import logging
import neat
import os
from pathlib import Path
import subprocess
import visualize

from api import deploy_endpoint

load_dotenv(override=True)
EXPORT_JAVA_HOME = "export JAVA_HOME=" + os.getenv("JAVA_HOME")
HOME_DIR = Path(os.getenv("GRAAL_REPO_DIR")) / "vm"
BENCHMARKS = ["akka-uct", "db-shootout", "dotty", "finagle-chirper", "finagle-http", "fj-kmeans", "future-genetic", "mnemonics", "par-mnemonics", "philosophers", "reactors", "rx-scrabble", "scala-doku", "scala-kmeans", "scala-stm-bench7", "scrabble"]
BENCHMARK_METRICS = ["time", "reachable-methods", "binary-size", "max-rss"]


def get_benchmark_cmd(benchmark):
    return "mx --env ni-ce benchmark \"renaissance-native-image:{}\"  --  --jvm=native-image --jvm-config=default-ce".format(benchmark)


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("NEAT evolution")


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # open api endpoint
    server = deploy_endpoint(net)

    # run benchmark, graal inliner uses endpoint running on `server.config.port`
    LOGGER.info(f"Launching benchmark {BENCHMARKS[0]}")
    completed_process = subprocess.run(
        f"{EXPORT_JAVA_HOME} && cd {HOME_DIR} && {get_benchmark_cmd(BENCHMARKS[0])}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    LOGGER.info(f"Benchmarking subprocess completed with {completed_process.returncode}")
    if completed_process.returncode != 0:
        LOGGER.info(f"Benchmarking subprocess completed with error:\n{completed_process.stderr}")

    stats = {}
    with open(Path(HOME_DIR) / "bench-results.json") as json_file:
        data = json.load(json_file)

        for q in data["queries"]:
            if q["metric.name"] in BENCHMARK_METRICS:
                stats[q["metric.name"]] = q["metric.value"]
    logging.debug(f"{genome} produced following benchmark stats: {stats}")

    if len(stats) != len(BENCHMARK_METRICS):
        return Exception(f"Some stats were not found in output. Expected {BENCHMARK_METRICS}, found {stats}.")

    # TODO experiment with different subgroups of collected metrics as fitness
    fitness = stats["reachable-methods"]

    # Delete endpoint after benchmarking is done
    server.close()
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
    winner = p.run(pe.evaluate, 10)

    LOGGER.info(f"Best genome: {winner}")

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'Inline'}
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config-feedforward'
    run(config_path)
