import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations

file_name = "data/infectious_hypergraph.dat"
beta = 1  # probability to infect
theta = 1  # minimum number of infectious nodes to start infection
MAX_TIME = 1392


def save_vectors(node_influence_vector,other_vector,vector_filename_save):
    # sorted_node_influence_vector = sorted(node_influence_vector)
    # sorted_other_vector = sorted(other_vector)
    with open(vector_filename_save,"w") as f:
        f.write('Influence vector: '+' '.join([str(node) for node in node_influence_vector])+'\n')
        f.write('Other vector:     '+' '.join([str(node) for node in other_vector])+'\n')

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


def get_all_nodes(hyperlinks_by_timestamp):
    nodes = set()
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                nodes.add(node)
    return nodes

def find_first_timestamp(hyperlinks_by_timestamp, seed):
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                if seed == node:
                    return timestamp


def infect(hyperlinks: list[list[int]], infected: set[int]):
    just_infected = set()
    for hyperlink in hyperlinks:
        infected_in_hyperlink = len(set(hyperlink).intersection(infected))
        if infected_in_hyperlink >= theta:
            for node in hyperlink:
                if node not in infected:
                    if will_infect():
                        just_infected.add(node)

    return just_infected


def get_ranked_influence(hyperlinks_by_timestamp: dict[int, list[list[int]]], nodes: list[int], infection_goal):
    x_percent_timestamp_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp = find_first_timestamp(hyperlinks_by_timestamp, seed)
        for timestamp in range(first_timestamp + 1, MAX_TIME):
            just_infected = infect(hyperlinks_by_timestamp[timestamp], infected_nodes)
            infected_nodes = just_infected.union(infected_nodes)
            if len(infected_nodes) >= infection_goal * len(nodes) and seed not in x_percent_timestamp_by_seed:
                x_percent_timestamp_by_seed[seed] = timestamp

    return dict(sorted(x_percent_timestamp_by_seed.items(), key=lambda item: item[1]))


def get_link_weight_vector(hyperlinks_by_timestamp: dict[int, list[list[int]]]):
    weight_by_link = {}
    for hyperlinks in hyperlinks_by_timestamp.values():
        for hyperlink in hyperlinks:
            links = list(combinations(sorted(hyperlink), 2))
            for link in links:
                e = tuple(sorted(link))
                if e in weight_by_link:
                    weight_by_link[e] += 1
                else:
                    weight_by_link[e] = 0
    return weight_by_link


def get_degree_vector(hyperlinks_by_timestamp: dict[int, list[list[int]]], nodes: list[int]):
    degree_by_node = {}
    visited_neighbours_by_node = {}
    for node in nodes:
        degree_by_node[node] = 0
        visited_neighbours_by_node[node] = set()
        for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
            for hyperlink in hyperlinks:
                if node in hyperlink:
                    for n in hyperlink:
                        if n != node and n not in visited_neighbours_by_node:
                            degree_by_node[node] += 1
                            visited_neighbours_by_node[node].add(n)
    return dict(sorted(degree_by_node.items(), key=lambda item: item[1], reverse=True))


def get_strength_vector(weight_by_link: dict[(int, int), int], nodes: list[int]):
    strength_by_node = {}
    for node in nodes:
        strength_by_node[node] = 0
        for (n1, n2), weight in weight_by_link.items():
            if n1 == node or n2 == node:
                strength_by_node[node] += weight
    return dict(sorted(strength_by_node.items(), key=lambda item: item[1], reverse=True))


def get_first_contact_timestamp(node: int, hyperlinks_by_timestamp: dict[int, list[list[int]]]):
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            if node in hyperlink:
                return timestamp


def get_first_contact_vector(hyperlinks_by_timestamp: dict[int, list[list[int]]], nodes: list[int]):
    first_contact_vector = {}
    for node in nodes:
        first_contact_vector[node] = get_first_contact_timestamp(node, hyperlinks_by_timestamp)
    # The earlier, the better
    return dict(sorted(first_contact_vector.items(), key=lambda item: item[1]))


# custom centrality metrics start
def get_importance_through_time(alpha=0.2):
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


def get_node_popularity(base_weight=1.1):
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

# custom centrality metrics end


def recognition_rate_RD_RS(degree_by_node: dict[int, int], strength_by_node: dict[int, int], x_percent_timestamp_by_node: dict[int, int],percentage_nodes = 0.1):
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ranks = list(x_percent_timestamp_by_node.keys())
    degrees = list(degree_by_node.keys())
    strengths = list(strength_by_node.keys())

    new_size = min(len(ranks), len(degrees), len(strengths))
    ranks = ranks[:new_size]
    degrees = degrees[:new_size]
    strengths = strengths[:new_size]
    assert len(ranks) == len(degrees) == len(strengths)


    rRD = []
    rRS = []
    for i in f:
        num_elements_to_extract = int(new_size * i)
        extracted_ranks = set(ranks[:num_elements_to_extract])
        denominator = len(extracted_ranks)

        numerator_D = len(extracted_ranks.intersection(set(degrees[:num_elements_to_extract])))
        numerator_S = len(extracted_ranks.intersection(set(strengths[:num_elements_to_extract])))

        rRD.append(numerator_D / denominator)
        rRS.append(numerator_S / denominator)

    plt.plot(f, rRD, marker='o', label='rRD')
    plt.plot(f, rRS, marker='s', label='rRS')
    plt.xlabel('Fraction according to nodes influence')
    plt.ylabel('Recognition Rate')
    plt.title('Recognition Rate for degree and strength VS fraction')
    plt.xticks(f)
    plt.legend()
    plt.savefig(f'ABOBA_10__{str(percentage_nodes*100)}.png')
    plt.show()


def recognition_rate_RD_RS_RF(degree_by_node: dict[int, int],
                              strength_by_node: dict[int, int],
                              first_contact_by_node: dict[int, int],
                              x_percent_timestamp_by_node: dict[int, int],
                              percentage_nodes = 0.1):
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ranks = list(x_percent_timestamp_by_node.keys())
    degrees = list(degree_by_node.keys())
    first_contact = list(first_contact_by_node.keys())
    strengths = list(strength_by_node.keys())

    new_size = min(len(ranks), len(degrees), len(strengths), len(first_contact))
    ranks = ranks[:new_size]
    degrees = degrees[:new_size]
    first_contact = first_contact[:new_size]
    strengths = strengths[:new_size]
    assert len(ranks) == len(degrees) == len(strengths) == len(first_contact)


    rRD = []
    rRS = []
    rRF = []
    for i in f:
        num_elements_to_extract = int(new_size * i)
        extracted_ranks = set(ranks[:num_elements_to_extract])
        denominator = len(extracted_ranks)

        numerator_D = len(extracted_ranks.intersection(set(degrees[:num_elements_to_extract])))
        numerator_S = len(extracted_ranks.intersection(set(strengths[:num_elements_to_extract])))
        numerator_F = len(extracted_ranks.intersection(set(first_contact[:num_elements_to_extract])))

        rRD.append(numerator_D / denominator)
        rRS.append(numerator_S / denominator)
        rRF.append(numerator_F / denominator)

    plt.plot(f, rRD, marker='o', label='rRD')
    plt.plot(f, rRS, marker='s', label='rRS')
    plt.plot(f, rRF, marker='^', label='rRF')
    plt.xlabel('Fraction according to nodes influence')
    plt.ylabel('Recognition Rate')
    plt.title('Recognition Rate for degree and strength and first contact VS fraction')
    plt.xticks(f)
    plt.legend()
    plt.savefig(f'ABOBA_11_{str(percentage_nodes*100)}.png')
    plt.show()

# recog. rate for weight through time and popularity through time
def recognition_rate_RT_RP(weight_through_time_by_node: dict[int, int],
                              popularity_through_time_by_node: dict[int, int],
                              x_percent_timestamp_by_node: dict[int, int],
                              percentage_nodes = 0.1):
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ranks = list(x_percent_timestamp_by_node.keys())
    weights = list(weight_through_time_by_node.keys())
    popularity = list(popularity_through_time_by_node.keys())

    new_size = min(len(ranks), len(popularity), len(weights))
    ranks = ranks[:new_size]
    weights = weights[:new_size]
    popularity = popularity[:new_size]
    assert len(ranks) == len(popularity) == len(weights)

    print(f'lengths: ranks: {len(ranks)}, popularity: {len(popularity)}, weights: {len(weights)}')

    rRT = []
    rRP = []
    for i in f:
        num_elements_to_extract = int(new_size * i)
        extracted_ranks = set(ranks[:num_elements_to_extract])
        denominator = len(extracted_ranks)

        numerator_T = len(extracted_ranks.intersection(set(weights[:num_elements_to_extract])))
        numerator_P = len(extracted_ranks.intersection(set(popularity[:num_elements_to_extract])))

        if (numerator_T / denominator) == 1.0:
            save_vectors(extracted_ranks,weights[:num_elements_to_extract],f'vectors_node_influence_and_weight_through_time_fraction_{str(i)}.txt')
        if (numerator_P / denominator) == 1.0:
            save_vectors(extracted_ranks,popularity[:num_elements_to_extract],f'vectors_node_influence_and_popularity_through_time_fraction_{str(i)}.txt')

        rRT.append(numerator_T / denominator)
        rRP.append(numerator_P / denominator)

    plt.plot(f, rRT, marker='o', label='rRT')
    for i, txt in enumerate(rRT):
        plt.annotate(f'{txt:.2f}', (f[i], rRT[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.plot(f, rRP, marker='s', label='rRP')
    for i, txt in enumerate(rRP):
        plt.annotate(f'{txt:.2f}', (f[i], rRP[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Fraction according to nodes influence')
    plt.ylabel('Recognition Rate')
    plt.title('Recognition Rate for weight through time (T) and \npopularity through time (P) VS fraction')
    plt.xticks(f)
    plt.legend()
    plt.savefig(f'ABOBA_12_{str(percentage_nodes*100)}.png')
    plt.show()

if __name__ == "__main__":
    hyperlinks_by_timestamp = read_file()
    nodes = get_all_nodes(hyperlinks_by_timestamp)
    percentage_nodes = 0.1

    x_percent_timestamp_by_seed = get_ranked_influence(hyperlinks_by_timestamp, list(nodes), percentage_nodes)
    weight_by_link = get_link_weight_vector(hyperlinks_by_timestamp)
    degree_by_node = get_degree_vector(hyperlinks_by_timestamp, list(nodes))
    strength_by_node = get_strength_vector(weight_by_link, list(nodes))
    first_contact_by_node = get_first_contact_vector(hyperlinks_by_timestamp, list(nodes))

    # our own centrality metrics (time importance and popularity)
    importance_through_time_by_node = get_importance_through_time(alpha=0.2)
    popularity_through_time_by_node = get_node_popularity(base_weight=1.1)


    # original rr plots
    recognition_rate_RD_RS(degree_by_node, strength_by_node, x_percent_timestamp_by_seed,percentage_nodes)
    recognition_rate_RD_RS_RF(degree_by_node, strength_by_node, first_contact_by_node, x_percent_timestamp_by_seed,percentage_nodes)

    # rr plots with new centrality metrics
    recognition_rate_RT_RP(importance_through_time_by_node,popularity_through_time_by_node,x_percent_timestamp_by_seed,percentage_nodes)