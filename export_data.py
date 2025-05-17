import csv
import json
import argparse


COLLECTING_REACHABLE_METHODS = True

parser = argparse.ArgumentParser(
                    prog='Data Exporter',
                    description='Export data about genome fitness from logs')
parser.add_argument('csv_file_path', help='Path to CSV containing training log')
args = parser.parse_args()

with open(args.csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    metrics = {}

    for index, row in enumerate(reader):
        if row.get('is_participating') == 'yes':
            content = row.get('content', 'No "content" column found')
            if row.get('group') == 'metrics':
                json_content = json.loads(content.replace("'", "\""))
                if COLLECTING_REACHABLE_METHODS:
                    metrics[row.get('match')] = json_content['reachable-methods']
                else:
                    metrics[row.get('match')] = json_content['binary-size']
            elif row.get('group') == 'benchmark_failed':
                metrics[row.get('match')] = 0
            elif row.get('group') == 'cached_genome':
                id_of_scenario = (int(row.get('match'))-1)%3
                offset_due_to_elitism = 2*3*int(max(int(content)-3, 0)/8) # calculate how many cached genomes have been encountered at this point
                starting_id_of_genome = (int(content)-1)*3 + 1 + offset_due_to_elitism
                metrics[row.get('match')] = metrics[str(starting_id_of_genome+id_of_scenario)]
            else:
                pass

    akka = []
    shootout = []
    dotty = []
    for i in range(1, 150, 3):
        akka.append(metrics[str(i)])
        shootout.append(metrics[str(i+1)])
        dotty.append(metrics[str(i+2)])


    def group_into_generations(metrics):
        return " ".join([str((i%10+1, metric/(1000 if COLLECTING_REACHABLE_METHODS else 1000000))) for i, metric in enumerate(metrics)])

    print("akka:")
    print(group_into_generations(akka))

    print("shootout:")
    print(group_into_generations(shootout))

    print("dotty:")
    print(group_into_generations(dotty))
