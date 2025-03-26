import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter

facebook = pd.read_csv("/Users/margheritatonon/applied_math_lab/networkx/facebook_combined.txt.gz", 
                       compression = "gzip", sep = " ", names = ["start_node", "end_node"])

G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")

plot_initial = False
if plot_initial == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos, width = 0.5, node_size = 10)
    plt.show()
    plt.close()

beta = 0.5 #"infection" rate
gamma = 0.3 #"recovery" rate

#giving ignorant to every node
state = []
nodes = G.nodes()
for node in nodes:
    state.append((node, "I"))
print(len(state))

random_node = np.random.choice(list(G.nodes))
print(f"the random spreader node is {random_node}")
state = [(node, "S") if node == random_node else (node, "I") for node, _ in state]
print(f"len state = {len(state)}")

#making a dictionary of the state
state_dict = dict(state)
print(f"len state_dict = {len(state_dict)}")

#updating the graph
nx.set_node_attributes(G, state_dict, "state")

ignorant_nodes = []
spreader_nodes = []
for key, label in state_dict.items():
    if label == "I":
        ignorant_nodes.append(key)
    else:
        spreader_nodes.append(key)
print(f"spreader node list = {spreader_nodes}")

first_updated_plot = False
if first_updated_plot == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos, nodelist = ignorant_nodes, node_color = "blue", node_size = 15)
    nx.draw(G, pos = pos, nodelist=spreader_nodes, node_color = "red", node_size = 15)
    plt.show()


#one iteration of the update rule:
state = nx.get_node_attributes(G, "state")
new_state = {}
for node in G.nodes():
    #ignorant nodes: for every spreader node they have in their vicinity, they have a probability beta of becoming spreaders themselves
    if state_dict[node] == "I":
        neighbors = list(G.neighbors(node))
        #now we access the state_dict 
        values = list(map(state_dict.get, neighbors))

        #we apply a map to the values so we know what the probabilities we have are
        def assign_values(my_list):
            mapping = {"S": 1, "I": 0}
            return list(map(mapping.get, my_list))
        num_s = assign_values(values) #so now we have a list with all of the probabilities of becoming infected for all of the neighbors of a current node

        #now we take a count of the nonzero values to see how many nonzero probabilities we have
        count_s = sum(num_s)
        #so then we need to do count_s times the random number between 0 and 1 in order to update the state_dict
        for n in range(count_s):
            if np.random.uniform(0, 1) < beta:
                new_state[node] = "S"
                break
        
        if "S" in values:
            print(f"S is in values! we are at node {node}")
            print(count_s)

state_dict.update(new_state)

ignorant_nodes = []
spreader_nodes = []
for key, label in state_dict.items():
    if label == "I":
        ignorant_nodes.append(key)
    else:
        spreader_nodes.append(key)

update_plot_two = True
if update_plot_two == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos, nodelist = ignorant_nodes, node_color = "blue", node_size = 15)
    nx.draw(G, pos = pos, nodelist=spreader_nodes, node_color = "red", node_size = 15)
    plt.show()
    plt.close()

print(state_dict)
value_counts = Counter(state_dict.values())
print(value_counts)