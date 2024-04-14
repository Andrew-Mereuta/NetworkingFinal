import numpy as np
import matplotlib.pyplot as plt
import random
import math

file_name = "data/infectious_hypergraph.dat"
beta = 0.6  # probability to infect
theta = 1  # minimum number of infectious nodes to start infection


def compute_node_weight_by_node_popularity(base_weight=1.1):
    """
        We create an importance metric where a node gains
        weight whenever the following condition is met:
        - A pair of nodes j,k is introduced by node i where node i 
        already had a connection before with j and k but not j and k.
            - The weight added weights more if the introduced nodes are
            of higher weight

        Note that whenever the two nodes j and k are "introduced" by i,
        a new connection is formed between them, therefore there is no
        need to keep track of introductions for later on since the original
        condition is already not satisfied by this observation.
    """
    graph_data_map = read_file()
    connection_information = {}  # keep track of known nodes by every other node
    weight_nodes = {}
    for timestep in range(len(graph_data_map)):
        for hyperlink in graph_data_map[timestep]:  # inspect all hyperlinks at timestep t
            for node in hyperlink:
                if node not in connection_information:
                    connection_information[node] = set()
                if node not in weight_nodes:
                    weight_nodes[node] = base_weight
                other_nodes = list(filter(lambda x: x != node and x in connection_information[node], hyperlink))
                added_pairs = set()
                for node_1 in other_nodes:
                    for node_2 in other_nodes:
                        if node_1 != node_2 or (node_1, node_2) in \
                                added_pairs or (node_2, node_1) in added_pairs:
                            # skip if same node or pair already explored in reverse order
                            continue
                        added_pairs.add((node_1, node_2))
                        if node_2 not in connection_information[node_1] and \
                                node_1 not in connection_information[node_2]:  # if nodes do not know each other
                            weight_node_1 = weight_nodes[node_1]
                            weight_node_2 = weight_nodes[node_2]
                            weight_nodes[node] += math.log(weight_node_1) + math.log(weight_node_2)
                            # make sure node_1 and node_2 now "know" each other
            for node in hyperlink:
                for node_1 in hyperlink:
                    if node_1 != node:
                        connection_information[node].add(node_1)
                        connection_information[node_1].add(node)
    return weight_nodes


def compute_importance_through_time(alpha=0.2):
    """
    This function computes the importance of nodes
    based on their number of activations that occurred
    on their links and the time at which those activations
    happened
    """
    graph_data_map = read_file()
    activation_nodes = {}
    weight_nodes = {}
    for timestep in range(len(graph_data_map)):
        occurring_nodes = set()
        # inspect all hyperlinks at timestep t
        for hyperlink in graph_data_map[timestep]:
            for node in hyperlink:
                # record all occurring nodes in the set of hyperlinks at timestep t for later computation of weight
                occurring_nodes.add(node)
                if node not in weight_nodes:
                    weight_nodes[node] = 0.0
                if node not in activation_nodes:
                    activation_nodes[node] = {}

                if timestep not in activation_nodes[node]:
                    activation_nodes[node][timestep] = []

                # add active links at that timestep, since each hyperlink is a clique, the num of links is num of nodes - 1
                activation_nodes[node][timestep].append(len(hyperlink) - 1)
        for node in occurring_nodes:
            active_nodes = sum(activation_nodes[node][timestep])  # sum up all the active edges of node at timestep t
            weight_nodes[node] += (active_nodes / (timestep + 1)) ** alpha

    return dict(sorted(weight_nodes.items(),key = lambda x: x[1],reverse=True))


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

        if infected_in_hyperlink >= theta:
            for node in hyperlink:
                if node not in infected:
                    if will_infect():
                        infected_nodes.add(node)

    return infected_nodes


def get_infected_nodes_by_timestamp(hyperlinks_by_timestamp: dict, nodes: list[int], infection_goal):
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


def calculate_nodes_by_degree(hyperlinks, nodes):
    degrees = {node: 0 for node in nodes}
    used_hyperlinks = set()
    for hyperlink in hyperlinks:
        if tuple(sorted(hyperlink)) not in used_hyperlinks:
            for node in hyperlink:
                degrees[node] += (len(hyperlink) - 1)
            used_hyperlinks.add(tuple(sorted(hyperlink)))

    sorted_grouped = {}
    degrees = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))
    for node, degree in sorted(degrees.items()):
        if degree in sorted_grouped:
            sorted_grouped[degree].append(node)
        else:
            sorted_grouped[degree] = [node]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0], reverse=True))


def calculate_nodes_by_weight(hyperlinks, nodes):
    weight_by_hyperlink = {}
    for hyperlink in hyperlinks:
        if tuple(sorted(hyperlink)) in weight_by_hyperlink:
            weight_by_hyperlink[tuple(sorted(hyperlink))] += (len(hyperlink) - 1)
        else:
            weight_by_hyperlink[tuple(sorted(hyperlink))] = 1

    node_strength = {node: 0 for node in nodes}
    for node in nodes:
        for hyperlink, weight in weight_by_hyperlink.items():
            if node in hyperlink:
                node_strength[node] += weight
    sorted_grouped = {}
    degrees = dict(sorted(node_strength.items(), key=lambda item: item[1], reverse=True))
    for node, strength in sorted(degrees.items()):
        if strength in sorted_grouped:
            sorted_grouped[strength].append(node)
        else:
            sorted_grouped[strength] = [node]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0], reverse=True))


def calculate_nodes_by_first_contact_timestamp(hyperlinks_by_timestamp):
    first_contact_timestamp_by_node = {}
    for (timestamp, hyperlinks) in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                if node in first_contact_timestamp_by_node:
                    first_contact_timestamp_by_node[node] = min(timestamp, first_contact_timestamp_by_node[node])
                else:
                    first_contact_timestamp_by_node[node] = timestamp
    first_contact_timestamp_by_node = dict(
        sorted(first_contact_timestamp_by_node.items(), key=lambda item: item[1], reverse=True))
    sorted_grouped = {}
    for node, timestamp in sorted(first_contact_timestamp_by_node.items()):
        if timestamp in sorted_grouped:
            sorted_grouped[timestamp].append(node)
        else:
            sorted_grouped[timestamp] = [node]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0]))


def centrality(nodes, hyperlinks, sorted_infected_nodes, hyperlinks_by_timestamp, num):
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    degrees = calculate_nodes_by_degree(hyperlinks, nodes)
    weights = calculate_nodes_by_weight(hyperlinks, nodes)
    fc = calculate_nodes_by_first_contact_timestamp(hyperlinks_by_timestamp)
    rRD_values = []
    rRS_values = []
    rFS_values = []
    for v in f:
        rRD = 0
        rRS = 0
        rFS = 0
        for _ in range(1000):
            Rf_degree = set()
            Rf_strength = set()
            Rf = set()
            Rf_contact = set()
            degreess = list(degrees.items())[0:int(v * len(nodes))]
            for key, value in degreess:
                Rf_degree.add(random.choice(value))
            weightss = list(weights.items())[0:int(v * len(nodes))]
            for key, value in weightss:
                Rf_strength.add(random.choice(value))
            sorted_infected_nodess = list(sorted_infected_nodes.items())[0:int(v * len(nodes))]
            for key, value in sorted_infected_nodess:
                Rf.add(random.choice(value))
            fcs = list(fc.items())[0:int(v * len(nodes))]
            for key, value in fcs:
                Rf_contact.add(random.choice(value))
            rRD += len(Rf.intersection(Rf_degree)) / len(Rf)
            rRS += len(Rf.intersection(Rf_strength)) / len(Rf)
            rFS += len(Rf.intersection(Rf_contact)) / len(Rf)
        rRD /= 1000
        rRS /= 1000
        rFS /= 1000
        rRD_values.append(rRD)
        rRS_values.append(rRS)
        rFS_values.append(rFS)

    plt.figure(figsize=(10, 6))
    plt.plot(f, rRD_values, marker='o', label='rRD')
    plt.plot(f, rRS_values, marker='s', label='rRS')
    if num == "10":
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for degree and strength vs fraction')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_10.png')
        plt.show()
    else:
        plt.plot(f, rFS_values, marker='^', label='r_first_contact')
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for degree, strength and first contact vs fraction')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_11.png')
        plt.show()


def plot_infected_nodes_by_seed(sorted_infected_nodes, assignment_num):
    sorted_infected_nodes = dict(sorted(sorted_infected_nodes.items()))
    seeds = list(sorted_infected_nodes.keys())
    timestamps = list(sorted_infected_nodes.values())

    plt.figure(figsize=(10, 6))
    plt.plot(seeds, timestamps, marker='o', color='b')
    plt.xlabel('Seed Nodes')
    plt.ylabel('Timestamp to Reach Goal')
    plt.title('Timestamp for Nodes to Reach Goal for Each Seed')
    plt.savefig(f"b_{assignment_num}.png")
    plt.show()


def get_networksb12(hyperlinks_by_timestamp: dict, nodes: list[int], infection_goal):
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp = find_first_timestamp(hyperlinks_by_timestamp, seed)
        t = 0  # flag
        for timestamp in range(first_timestamp, len(hyperlinks_by_timestamp.keys())):
            infected_ns = infect(hyperlinks_by_timestamp[timestamp], infected_nodes)
            inf_l = infected_ns - infected_nodes
            infected_nodes = infected_ns.union(infected_nodes)

            if len(infected_nodes) <= infection_goal * len(nodes):
                for _ in range(len(inf_l)):
                    if seed in infected_nodes_by_seed:
                        infected_nodes_by_seed[seed].append(timestamp)
                    else:
                        infected_nodes_by_seed[seed] = [timestamp]
            else:
                if t == 0:
                    for s in range(len(inf_l)):
                        if seed in infected_nodes_by_seed:
                            infected_nodes_by_seed[seed].append(timestamp)
                        else:
                            infected_nodes_by_seed[seed] = [timestamp]
                    t += 1

    average_times = {}
    for seed, times in infected_nodes_by_seed.items():
        average_times[seed] = sum(times) / len(times)

    sorted_grouped = {}
    for key, val in sorted(average_times.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0]))


def b12(sorted_infected_nodes_r, sorted_infected_nodes_r_star, sorted_infected_nodes_r_accent, num, nodes):
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    rRD_values = []
    rRS_values = []
    rFS_values = []
    for v in f:
        rRD = 0
        rRS = 0
        rFS = 0
        for _ in range(1000):
            Rf_star = set()
            Rf_accent = set()
            Rf = set()
            degreess = list(sorted_infected_nodes_r.items())[0:int(v * len(nodes))]
            for key, value in degreess:
                Rf.add(random.choice(value))
            weightss = list(sorted_infected_nodes_r_star.items())[0:int(v * len(nodes))]
            for key, value in weightss:
                Rf_star.add(random.choice(value))
            sorted_infected_nodess = list(sorted_infected_nodes_r_accent.items())[0:int(v * len(nodes))]
            for key, value in sorted_infected_nodess:
                Rf_accent.add(random.choice(value))
            rRD += len(Rf.intersection(Rf_star)) / len(Rf)
            rRS += len(Rf.intersection(Rf_accent)) / len(Rf)
        rRD /= 1000
        rRS /= 1000
        rFS /= 1000
        rRD_values.append(rRD)
        rRS_values.append(rRS)
        rFS_values.append(rFS)

    plt.figure(figsize=(10, 6))
    if num == '1':
        plt.plot(f, rRD_values, marker='o', label='r*')
        plt.plot(f, rRS_values, marker='s', label='r\'')
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for r* and r\' with regards to r')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_12_1.png')
        plt.show()
    if num == '2':
        plt.plot(f, rRD_values, marker='o', label='r')
        plt.plot(f, rRS_values, marker='s', label='r*')
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for r and r* with regards to r\'')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_12_2.png')
        plt.show()
    if num == '3':
        plt.plot(f, rRD_values, marker='o', label='r')
        plt.plot(f, rRS_values, marker='s', label='r\'')
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for r and r\' with regards to r*')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_12_3.png')
        plt.show()


def plot_weight_by_node(weight_by_node):
    # Extracting nodes and weights
    nodes = list(weight_by_node.keys())
    weights = list(weight_by_node.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, weights, color='b', linestyle='-')
    plt.xlabel('Node')
    plt.ylabel('Weight')
    plt.title('Weight by Node')
    plt.savefig('b_adrian.png')
    plt.show()



if __name__ == "__main__":
    hyperlinks_by_timestamp = read_file()
    nodes = get_all_nodes(hyperlinks_by_timestamp)
    # WTF is not_map??? @Anna
    infected_nodes_by_timestamp, sorted_infected_nodes, not_map = get_infected_nodes_by_timestamp(
        hyperlinks_by_timestamp, list(nodes), 0.8)

    # plot_average_infected_with_error_bars(infected_nodes_by_timestamp)

    plot_infected_nodes_by_seed(not_map, "9")

    # hyperlinks = [inner for outer in hyperlinks_by_timestamp.values() for inner in outer]
    #
    # centrality(nodes, hyperlinks, sorted_infected_nodes, hyperlinks_by_timestamp, "10")
    #
    # centrality(nodes, hyperlinks, sorted_infected_nodes, hyperlinks_by_timestamp, "11")
    #
    # infected_nodes_by_timestamp_r_star, sorted_infected_nodes_r_star, not_map_r_star = get_infected_nodes_by_timestamp(
    #     hyperlinks_by_timestamp, list(nodes), 0.1)
    # sorted_infected_nodes_r_accent = get_networksb12(hyperlinks_by_timestamp, list(nodes), 0.8)
    # b12(sorted_infected_nodes, sorted_infected_nodes_r_star, sorted_infected_nodes_r_accent, '1', nodes)
    # b12(sorted_infected_nodes_r_accent, sorted_infected_nodes, sorted_infected_nodes_r_star, '2', nodes)
    # b12(sorted_infected_nodes_r_star, sorted_infected_nodes, sorted_infected_nodes_r_accent, '3', nodes)
    #
    # plot_weight_by_node(compute_importance_through_time())
