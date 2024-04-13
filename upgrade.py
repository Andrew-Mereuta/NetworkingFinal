import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations

file_name = "data/infectious_hypergraph.dat"
beta = 1  # probability to infect
theta = 1  # minimum number of infectious nodes to start infection
MAX_TIME = 1392

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


def recognition_rate_RD_RS(degree_by_node: dict[int, int], strength_by_node: dict[int, int], x_percent_timestamp_by_node: dict[int, int]):
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
    plt.savefig('ABOBA_10.png')
    plt.show()


def recognition_rate_RD_RS_RF(degree_by_node: dict[int, int],
                              strength_by_node: dict[int, int],
                              first_contact_by_node: dict[int, int],
                              x_percent_timestamp_by_node: dict[int, int]):
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
    plt.savefig('ABOBA_11.png')
    plt.show()


if __name__ == "__main__":
    hyperlinks_by_timestamp = read_file()
    nodes = get_all_nodes(hyperlinks_by_timestamp)

    x_percent_timestamp_by_seed = get_ranked_influence(hyperlinks_by_timestamp, list(nodes), 0.1)
    weight_by_link = get_link_weight_vector(hyperlinks_by_timestamp)
    degree_by_node = get_degree_vector(hyperlinks_by_timestamp, list(nodes))
    strength_by_node = get_strength_vector(weight_by_link, list(nodes))
    first_contact_by_node = get_first_contact_vector(hyperlinks_by_timestamp, list(nodes))

    recognition_rate_RD_RS(degree_by_node, strength_by_node, x_percent_timestamp_by_seed)
    recognition_rate_RD_RS_RF(degree_by_node, strength_by_node, first_contact_by_node, x_percent_timestamp_by_seed)