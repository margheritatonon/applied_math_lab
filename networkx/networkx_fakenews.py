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

#this assigns the initial state
ignorant_nodes = []
spreader_nodes = []
for key, label in state_dict.items():
    if label == "I":
        ignorant_nodes.append(key)
    else:
        spreader_nodes.append(key)
print(f"spreader node list = {spreader_nodes}")

first_updated_plot = False #this is the plot after 1 person is assigned the fake news
if first_updated_plot == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos, nodelist = ignorant_nodes, node_color = "blue", node_size = 15)
    nx.draw(G, pos = pos, nodelist=spreader_nodes, node_color = "red", node_size = 15)
    plt.show()


#one iteration of the update rule:
def update_rule(G):
    state = nx.get_node_attributes(G, "state")
    new_state = {}

    def assign_values(my_list):
        mapping = {"S": 1, "I": 0, "R":0}
        return list(map(mapping.get, my_list))
    
    for node in G.nodes():
        #ignorant nodes: for every spreader node they have in their vicinity, they have a probability beta of becoming spreaders themselves
        if state_dict[node] == "I":
            neighbors = list(G.neighbors(node))
            #now we access the state_dict 
            values = list(map(state_dict.get, neighbors))

            #we apply a map to the values so we know what the probabilities we have are
            num_s = assign_values(values) #so now we have a list with all of the probabilities of becoming infected for all of the neighbors of a current node

            #now we take a count of the nonzero values to see how many nonzero probabilities we have
            #create an array with as many random numbers as susceptible neighbors and see if any of them are less than beta
            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < beta):
                new_state[node] = "S"

        
        #spreader node: has a probability gamma of becoming a stifler for every spreader it has in its neighborhood
        if state_dict[node] == "S" or state_dict[node] == "R":
            neighbors = list(G.neighbors(node))
            values = list(map(state_dict.get, neighbors))
            #assign values so we know what probabilities we have
            num_s = assign_values(values)

            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < gamma):
                new_state[node] = "R"

    return new_state

pos = nx.spring_layout(G)
all_states = []
num_iters = 3
for i in range(num_iters):
    new_state = update_rule(G)
    all_states.append(new_state)
    state_dict.update(new_state)

#print(all_states)


"""
ignorant_nodes = []
spreader_nodes = []
rstifler_nodes = []
for key, label in state_dict.items():
    if label == "I":
        ignorant_nodes.append(key)
    elif label == "S":
        spreader_nodes.append(key)
    else:
        rstifler_nodes.append(key)

update_plot_two = True
if update_plot_two == True:
    nx.draw(G, pos = pos, nodelist = ignorant_nodes, node_color = "blue", node_size = 15)
    nx.draw(G, pos = pos, nodelist=spreader_nodes, node_color = "red", node_size = 15)
    nx.draw(G, pos = pos, nodelist=rstifler_nodes, node_color = "green", node_size = 15)
    plt.title(f"Iteration {i}")
    plt.show()
    plt.close()

print(state_dict)
value_counts = Counter(state_dict.values())
print(value_counts)"""