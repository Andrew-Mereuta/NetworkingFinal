import numpy as np
from matplotlib import pyplot as plt

file_name = "data/infectious_hypergraph.dat"
def read_file():
    with open(file_name, "r") as file:
        data = file.readlines()
    parsed_data_map = {}
    for i, line in enumerate(data):
        parsed_data_map[i] = eval(line.strip())

    return parsed_data_map

def calculate_activity(parsed_data):
    activity = {}
    for i in range(0, 500):
        current_collab = parsed_data[i]
        for curr in current_collab:
            for j in curr:
                activity.setdefault(j, 0)
                activity[j] += len(curr)
    return activity

def calculate_correlation_based_on_static_activity(parsed_data, activity):
    later_activity = {}
    for i in range(500, len(parsed_data)):
        current_collab = parsed_data[i]
        for curr in current_collab:
            for j in curr:
                later_activity.setdefault(j, 0)
                later_activity[j] += 1
                if j not in activity:
                    activity[j] = 0
    for j in activity:
        if j not in later_activity:
            later_activity[j] = 0
    sorted_activity = dict(sorted(activity.items()))
    sorted_later_activity = dict(sorted(later_activity.items()))
    initial_activity_values = np.array(list(sorted_activity.values()))
    later_activity_values = np.array(list(sorted_later_activity.values()))
    correlation = np.corrcoef(initial_activity_values, later_activity_values)[0, 1]

    return correlation

def calculate_correlation_with_step(parsed_data):
    correlations = {}
    for m in range(0, len(parsed_data)):
        for step in range(5, int((len(parsed_data) - m)/2), 5):
            activity = {}
            for i in range(0, step):
                current_collab = parsed_data[i]
                for curr in current_collab:
                    for j in curr:
                        activity.setdefault(j, 0)
                        activity[j] += len(curr)
            later_activity = {}
            for i in range(step, 2 * step):
                current_collab = parsed_data[i]
                for curr in current_collab:
                    for j in curr:
                        later_activity.setdefault(j, 0)
                        later_activity[j] += 1
                        if j not in activity:
                            activity[j] = 0
            for j in activity:
                if j not in later_activity:
                    later_activity[j] = 0
            sorted_activity = dict(sorted(activity.items()))
            sorted_later_activity = dict(sorted(later_activity.items()))
            initial_activity_values = np.array(list(sorted_activity.values()))
            later_activity_values = np.array(list(sorted_later_activity.values()))
            correlation = np.corrcoef(initial_activity_values, later_activity_values)[0, 1]
            correlations.setdefault(m, [])
            correlations[m].append(correlation)
    avg = []
    num_nodes = []
    for i in range(0, len(correlations[0])):
        avg_correlations = []
        for m in correlations:
            if len(correlations[m]) > i:
                avg_correlations.append(correlations[m][i])
            else:
                break
        avg.append(np.mean(avg_correlations))
        num_nodes.append(5 * (i+1))
    plt.figure(figsize=(8, 5))
    plt.plot(num_nodes, avg, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Nodes Considered')
    plt.ylabel('Average Correlation')
    plt.title('Average Correlations vs. Number of Nodes Considered')
    plt.show()
    plt.savefig('correlations.png')
    return avg

if __name__ == "__main__":
    parsed_data = read_file()
    # static activity
    static_activity = calculate_activity(parsed_data)
    # this is negative as nodes that were active at teh beginning are not active anymore
    # general_correlation = calculate_correlation_based_on_static_activity(parsed_data, static_activity)
    print(calculate_correlation_with_step(parsed_data))


# [[0,1]]  - what is the probability that if [n1, n2] [n2, n3] -> node will be connected in the future