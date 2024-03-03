from dotenv import load_dotenv
import json
import logging
import multiprocessing
import neat
import os
from pathlib import Path
import subprocess
import visualize

from api import launch_service

load_dotenv()
EXPORT_JAVA_HOME = "export JAVA_HOME=" + os.getenv("JAVA_HOME")
HOME_DIR = Path(os.getenv("GRAAL_REPO_DIR")) / "vm"
BENCHMARKS = ["akka-uct", "db-shootout", "dotty", "finagle-chirper", "finagle-http", "fj-kmeans", "future-genetic", "mnemonics", "par-mnemonics", "philosophers", "reactors", "rx-scrabble", "scala-doku", "scala-kmeans", "scala-stm-bench7", "scrabble"]
BENCHMARK_METRICS = ["time", "reachable-methods", "binary-size", "max-rss"]


def get_benchmark_cmd(benchmark):
    return "mx --env ni-ce benchmark \"renaissance-native-image:{}\"  --  --jvm=native-image --jvm-config=default-ce".format(benchmark)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("NEAT evolution")


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # open api endpoint
    _ = launch_service(net)

    # run benchmark, graal inliner uses endpoint running on `port`
    try:
        completed_process = subprocess.run(
            f"{EXPORT_JAVA_HOME} && cd {HOME_DIR} && {get_benchmark_cmd(BENCHMARKS[0])}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        if completed_process.returncode != 0:
            return Exception(f"Error occurred: {completed_process.stderr}")
    except subprocess.CalledProcessError as e:
        return f"Error occurred: {e}"

    stats = {}
    with open(Path(HOME_DIR) / "bench-results.json") as json_file:
        data = json.load(json_file)

        for q in data["queries"]:
            if q["metric.name"] in BENCHMARK_METRICS:
                stats[q["metric.name"]] = q["metric.value"]
    logging.debug(f"{genome} produced following benchmark stats: {stats}")

    if len(stats) != len(BENCHMARK_METRICS):
        return Exception(f"Some stats were not found in output. Expected {BENCHMARK_METRICS}, found {stats}.")

    # use inverted time as fitness
    # TODO experiment with different subgroups of collected metrics as fitness
    fitness = -stats["time"]
    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    logger.info("Initial population has been created.")

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    pe = neat.ParallelEvaluator(1, eval_genome)

    logger.info("Running simulation.")
    winner = p.run(pe.evaluate, 10)

    logger.info(f"Best genome: {winner}")

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'Inline'}
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config-feedforward'
    run(config_path)
