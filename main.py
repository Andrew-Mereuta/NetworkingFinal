import numpy as np
import matplotlib.pyplot as plt
import random

file_name = "data/infectious_hypergraph.dat"
beta = 0.6  # probability to infect
theta = 0.3  # minimum percentage of infectious nodes to start infection


def read_file():
    with open(file_name, "r") as file:
        data = file.readlines()
    parsed_data_map = {}
    for i, line in enumerate(data):
        parsed_data_map[i] = eval(line.strip())

    return parsed_data_map


def will_infect():
    choices = [True, False]
    probabilities = [beta, 1 - beta]
    return random.choices(choices, weights=probabilities)[0]


def calculate_average_infected(infected_nodes_by_timestamp):
    average_infected = {}
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        num_networks = len(num_infected_list)
        total_infected = sum(num_infected_list)
        average_infected[timestamp] = total_infected / num_networks
    return average_infected


def calculate_standard_deviation(infected_nodes_by_timestamp):
    std_deviation_by_timestamp = {}
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        std_deviation = np.std(num_infected_list)
        std_deviation_by_timestamp[timestamp] = std_deviation
    return std_deviation_by_timestamp


def plot_average_infected_with_error_bars(infected_nodes_by_timestamp):
    timestamps = list(infected_nodes_by_timestamp.keys())
    averages = []
    std_deviations = []
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        average = calculate_average_infected(infected_nodes_by_timestamp)
        std_deviation = np.std(num_infected_list)
        averages.append(average)
        std_deviations.append(std_deviation)
    averages = list(calculate_average_infected(infected_nodes_by_timestamp).values())
    std_deviations = list(calculate_standard_deviation(infected_nodes_by_timestamp).values())
    plt.errorbar(timestamps, averages, yerr=std_deviations)
    plt.xlabel('Timestamp')
    plt.ylabel('Average Number of Infected Nodes (E[I(t)])')
    plt.title('Average Number of Infected Nodes Over Time')
    plt.savefig('b_8.png')
    plt.show()


def find_first_timestamp(hyperlinks_by_timestamp, seed):
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                if node == seed:
                    return timestamp


def infect(hyperlinks: [[int]], infected: set):
    infected_nodes = infected.copy()

    for hyperlink in hyperlinks:
        infected_in_hyperlink = len(set(hyperlink).intersection(infected))
        infected_percentage = infected_in_hyperlink / (len(hyperlink))

        if infected_percentage >= theta:
            for node in hyperlink:
                if node not in infected:
                    if will_infect():
                        infected_nodes.add(node)

    return infected_nodes


def get_networks(hyperlinks_by_timestamp: dict, nodes: list[int], infection_goal):
    infected_nodes_by_timestamp = {}
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp = find_first_timestamp(hyperlinks_by_timestamp, seed)
        for timestamp in range(first_timestamp, len(hyperlinks_by_timestamp.keys())):
            infected_ns = infect(hyperlinks_by_timestamp[timestamp], infected_nodes)
            infected_nodes = infected_ns.union(infected_nodes)
            if timestamp in infected_nodes_by_timestamp:
                infected_nodes_by_timestamp[timestamp].append(len(infected_nodes))
            else:
                infected_nodes_by_timestamp[timestamp] = [len(infected_nodes)]
            if seed not in infected_nodes_by_seed:
                if len(infected_nodes) > infection_goal * len(nodes):
                    infected_nodes_by_seed[seed] = timestamp

    sorted_grouped = {}
    for seed, infected_nodes in sorted(infected_nodes_by_seed.items()):
        if infected_nodes in sorted_grouped:
            sorted_grouped[infected_nodes].append(seed)
        else:
            sorted_grouped[infected_nodes] = [seed]
    return infected_nodes_by_timestamp, dict(sorted(sorted_grouped.items(), key=lambda x: x[0])), dict(
        sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))


def get_all_nodes(hyperlinks_by_timestamp):
    nodes = set()
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                nodes.add(node)
    return nodes


if __name__ == "__main__":
    hyperlinks_by_timestamp = read_file()
    nodes = get_all_nodes(hyperlinks_by_timestamp)
    # WTF is not map??? @Anna
    infected_nodes_by_timestamp, sorted_infected_nodes, not_map = get_networks(hyperlinks_by_timestamp, list(nodes), 0.8)

    plot_average_infected_with_error_bars(infected_nodes_by_timestamp)