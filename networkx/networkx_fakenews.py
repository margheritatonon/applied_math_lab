import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.animation as animation

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
            else:
                if state_dict[node] == "S":
                    new_state[node] = "S"
                else:
                    new_state[node] = "R"

    return new_state

pos = nx.spring_layout(G)
all_states = []
num_iters = 10
for i in range(num_iters):
    new_state = update_rule(G)
    all_states.append(new_state)
    state_dict.update(new_state)

#print(all_states)

time_arr = np.linspace(0, num_iters, num_iters) #x axis
rss = [] #y
iss = [] #y
sss = [] #y
for st in all_states:
    value_counts = Counter(st.values())
    print(value_counts) #the other ones that are not present here are I
    #4039 nodes in total

    #making sure it doesnt cause an error if there are no R or S nodes
    try:
        count_r = value_counts["R"]
    except:
        count_r = 0
    rss.append(count_r)

    try:
        count_s = value_counts["S"]
    except:
        count_s = 0
    sss.append(count_s)

    #calculaging the number of ignorant (not present in new_state)
    total_count_rs = count_r + count_s 
    count_i = 4039 - total_count_rs
    iss.append(count_i)

#now we can plot the graph of these versus iterations
static = False
if static == True:
    fig, ax = plt.subplots()
    ax.plot(time_arr, np.array(iss), label = "I", color = "blue")
    ax.plot(time_arr, np.array(rss), label = "R", color = "green")
    ax.plot(time_arr, np.array(sss), label = "S", color = "red")
    ax.legend()
    ax.set_title("Ignorant (I), Spreader (S), and Stifler (R) Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    plt.show()


#creating subplots:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
(plot_graph,) = ax1.plot([], []) #for the actual graph
(plot_overtime,) = ax2.plot([], []) #for the plot ov SIR over time

#for the animation:
#need to at every time step save the network state but also the number of S, I, R for the other plot

def animate_SIR(i, x, ignorant, spreader, stifler): #so i need something that returns ignorant, spreader, stifler numbers in total, like over the entire simulation life
    plot_overtime.set_data(x[:i], ignorant[:i], color = "blue", label = "Ignorant") #x needs to be the time array, y needs to be the SIR... but we have 3 of them so idk how
    plot_overtime.set_data(x[:i], spreader[:i], color = "red", label = "Spreader")
    plot_overtime.set_data(x[:i], stifler[:i], color = "green", label = "Stifler")
    return plot_overtime,


ani1 = animation.FuncAnimation(fig, animate_SIR, fargs=(time_arr, iss, sss, rss), interval=70, blit=False)
ax1.legend()
ax1.set_title("Ignorant (I), Spreader (S), and Stifler (R) Evolution")
ax1.set_xlabel("Time")
ax1.set_ylabel("Population")

#ax1.plot(xx_nullcline_pts_sorted, xy_nullcline_pts_sorted, color = "red", label = "dx/dt nullcline", ls = ":")
#ax1.plot(yx_nullcline_pts_sorted, yy_nullcline_pts_sorted, color = "green", label = "dy/dt nullcine", ls = ":")
#ax1.scatter(fsolve_res[0], fsolve_res[1], color = "black", label = "fixed point")
#ax1.set_xlabel("x")
#ax1.set_ylabel("y")
#ax1.legend()
#ax1.set_title("Van der Pol Phase Plane")

plt.tight_layout()
plt.show()

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