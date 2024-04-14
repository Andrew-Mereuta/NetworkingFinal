file_name = "data/infectious_hypergraph.dat"

def read_file():
    with open(file_name, "r") as file:
        data = file.readlines()
    parsed_data_map = {}
    for i, line in enumerate(data):
        parsed_data_map[i] = eval(line.strip())

    return parsed_data_map


def produce_pariwise_dataset():
    data = read_file()
    pairwise_data = {}
    with open("./data/pairwise_dataset.txt","w") as f:
        for timestep in data:
            pairwise_data[timestep] = []
            added_pairs = set()
            f.write('[') # open bracket of line for timestep in dataset
            for i,hyperlink in enumerate(data[timestep]):
                for j,node_1 in enumerate(hyperlink):
                    for node_2 in hyperlink:
                        sorted_tuple = tuple(sorted((node_1,node_2))) # to keep track of already added connections for that timestep
                        if node_1 != node_2 and sorted_tuple not in added_pairs:
                            added_pairs.add(sorted_tuple)
                            if i == 0 and j == 0:
                                f.write(f'[{node_1},{node_2}]')
                            else:
                                f.write(f', [{node_1},{node_2}]')
                            pairwise_data[timestep].append(list(sorted_tuple))
            f.write(']\n')

if __name__ == "__main__":
    produce_pariwise_dataset()