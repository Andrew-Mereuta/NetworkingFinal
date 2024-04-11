# node influence - if there are couple of groups, and they connect through that node
import numpy as np
from matplotlib import pyplot as plt

file_name = "data/infectious_hypergraph.dat"


def read_file(file_name):
    with open(file_name, "r") as file:
        data = file.readlines()

    parsed_data_map = {}
    for i, line in enumerate(data):
        # Remove leading/trailing whitespace and evaluate the line as a list of lists
        parsed_data_map[i] = eval(line.strip())

    return parsed_data_map

def get_all_nodes(hyperlinks_by_timestamp):
    nodes = set()
    for timestamp, hyperlinks in hyperlinks_by_timestamp.items():
        for hyperlink in hyperlinks:
            for node in hyperlink:
                nodes.add(node)
    return nodes


def node_influence(nodes, network_parse):
    node_map = {}
    for node in nodes:
        for timestamp in network_parse:
            for n in network_parse[timestamp]:
                if node in n:
                    node_map.setdefault(node, [])
                    node_map[node].append((timestamp, set(n) - {node}))

    return node_map

def all_perm(node_influence):
    influence = {}
    for i in node_influence:
        checked_pairs = []
        for index in range(0, len(node_influence[i])-1):
            timestamp1, interaction1 = node_influence[i][index]
            for ind in range(0, len(node_influence[i])):
                timestamp2, interaction2 = node_influence[i][ind]
                if (interaction1, interaction2) in checked_pairs:
                    continue
                checked_pairs.append((interaction1, interaction2))
                # interactions = node_influence(set(interaction1).union(set(interaction2)))
                for n in interaction1:
                    interaction = node_influence[n]
                    for (t, inter) in interaction:
                        if t >= timestamp1 and t >= timestamp2:
                            if interaction2.issubset(inter):
                                influence.setdefault(i, 0)
                                influence[i] += 1
                                break
    return dict(sorted(influence.items(), key=lambda x: x[1], reverse=True))

def all_perm_weighted(node_influence):
    influence = {}
    for i in node_influence:
        checked_pairs = []
        for index in range(0, len(node_influence[i])-1):
            timestamp1, interaction1 = node_influence[i][index]
            for ind in range(0, len(node_influence[i])):
                timestamp2, interaction2 = node_influence[i][ind]
                if (interaction1, interaction2) in checked_pairs:
                    continue
                checked_pairs.append((interaction1, interaction2))
                # interactions = node_influence(set(interaction1).union(set(interaction2)))
                for n in interaction1:
                    interaction = node_influence[n]
                    for (t, inter) in interaction:
                        if t >= timestamp1 and t >= timestamp2:
                            if interaction2.issubset(inter):
                                influence.setdefault(i, 0)
                                influence[i] += 1 * timestamp2/t
                                break
    return dict(sorted(influence.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    data_map = read_file(file_name)
    my_node_influence = node_influence(get_all_nodes(data_map), data_map)
    print(all_perm(my_node_influence))
    print(all_perm_weighted(my_node_influence))



