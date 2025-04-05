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

beta = 0.8 #"infection" rate
gamma = 0.3 #"recovery" rate


#giving ignorant to every node
state = []
nodes = G.nodes()
for node in nodes:
    state.append((node, "I"))
print(len(state))

#here we can choose whether we want 1 node initially or if we want 5 nodes (possible interaction)
one_node = True
if one_node == True:
    random_node = np.random.choice(list(G.nodes))
    print(f"the random spreader node is {random_node}")
    state = [(node, "S") if node == random_node else (node, "I") for node, _ in state]
    print(f"len state = {len(state)}")
else:
    random_nodes = np.random.choice(list(G.nodes), size = 5)
    print(f"the random spreader nodes are {random_nodes}")
    state = [(node, "S") if node in random_nodes else (node, "I") for node, _ in state]

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
    """
    Returns the new state of a graph given the current graph.
    """
    state = nx.get_node_attributes(G, "state")
    new_state = {}

    def assign_values(my_list):
                mapping = {"S": 1, "I": 0, "R":0}
                return list(map(mapping.get, my_list))
    
    for node in G.nodes():
        #ignorant nodes: for every spreader node they have in their vicinity, they have a probability beta of becoming spreaders themselves
        if state[node] == "I":
            neighbors = list(G.neighbors(node))
            #now we access the state_dict 
            values = list(map(state.get, neighbors))

            #we apply a map to the values so we know what the probabilities we have are
            num_s = assign_values(values) #so now we have a list with all of the probabilities of becoming infected for all of the neighbors of a current node

            #now we take a count of the nonzero values to see how many nonzero probabilities we have
            #create an array with as many random numbers as susceptible neighbors and see if any of them are less than beta
            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < beta):
                new_state[node] = "S"

        
        #spreader node: has a probability gamma of becoming a stifler for every spreader it has in its neighborhood
        if state[node] == "S":
            neighbors = list(G.neighbors(node))
            values = list(map(state.get, neighbors))
            num_s = assign_values(values)

            rand_arr = np.random.uniform(0, 1, sum(num_s)) #creating an array of random variables based on how many stiflers there are
            if np.any(rand_arr < gamma):
                new_state[node] = "R"
            else:
                new_state[node] = "S"
        elif state[node] == "R":
            new_state[node] = "R"

    return new_state

num_iters = 20

def all_iters(num_iters):
    """
    Returns all of the states of the graph for the given number of iterations.
    """
    pos = nx.spring_layout(G)
    all_states = []
    for i in range(num_iters):
        new_state = update_rule(G)
        all_states.append(new_state)
        state_dict.update(new_state)
        nx.set_node_attributes(G, state_dict, "state")
    return all_states

time_arr = np.linspace(0, num_iters, num_iters) #x axis

all_states = all_iters(num_iters)

def extract_sir(all_states):
    """
    Returns a tuple of the number of ignorant, spreader, and stifler nodes for all states for all iterations
    """
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

        #calculating the number of ignorant (not present in new_state)
        total_count_rs = count_r + count_s 
        count_i = 4039 - total_count_rs
        iss.append(count_i)
    return rss, iss, sss 

rss, iss, sss = extract_sir(all_states)

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
line_I, = ax2.plot([], [], color="blue", label="Ignorant")
line_S, = ax2.plot([], [], color="red", label="Spreader")
line_R, = ax2.plot([], [], color="green", label="Stifler") #for the plot ov SIR over time


#NETWORK ANIMATION:
#defining the graph
pos = nx.spring_layout(G)

#we care about all_iters --> return a list of all of the node numbers and their states for every iteration
#defining the animation function
#all_states = all_iters(num_iters) --> we already have this defined above

#nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2) #drawing the edges
"""def animate_graph(i, G, pos):
    ax1.clear()
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2)
    curr_state = all_states[i] #this is the node number and the state it is

    #need to extract the S, I, R nodes from this dictionary
    i_list = list(map(lambda keyval: keyval[0], filter(lambda keyval: keyval[1] == "I", curr_state.items())))
    s_list = list(map(lambda keyval: keyval[0], filter(lambda keyval: keyval[1] == "S", curr_state.items())))
    r_list = list(map(lambda keyval: keyval[0], filter(lambda keyval: keyval[1] == "R", curr_state.items())))
    
    #drawing
    nx.draw_networkx_nodes(G, pos = pos, nodelist = i_list, node_color = "blue", node_size = 15, ax=ax1)
    nx.draw_networkx_nodes(G, pos = pos, nodelist=s_list, node_color = "red", node_size = 15, ax = ax1)
    nx.draw_networkx_nodes(G, pos = pos, nodelist = r_list, node_color = "green", node_size = 15, ax=ax1)   
    
    return ax1.collections

ani1 = animation.FuncAnimation(fig, animate_graph, fargs = (G, pos), interval = 70, blit = True, frames=num_iters)

#SIR PLOT ANIMATION
def animate_SIR(i, x, ignorant, spreader, stifler):
    line_I.set_data(x[:i], ignorant[:i]) #it is complaining about unexpected argument color here
    line_S.set_data(x[:i], spreader[:i])
    line_R.set_data(x[:i], stifler[:i])
    return line_I, line_S, line_R,

ax2.set_xlim(0, num_iters)
ax2.set_ylim(0, 4100)

ani2 = animation.FuncAnimation(fig, animate_SIR, fargs=(time_arr, iss, sss, rss), interval=70, blit=True, frames=num_iters)
ax2.legend()
ax2.set_title("Ignorant (I), Spreader (S), and Stifler (R) Evolution")
ax2.set_xlabel("Time")
ax2.set_ylabel("Population")"""

#ANIMATION OF BOTH:
def combined_animation(i, G, pos, x, iss, sss, rss):
    curr_state = all_states[i]
    i_list = [n for n, st in curr_state.items() if st == "I"]
    s_list = [n for n, st in curr_state.items() if st == "S"]
    r_list = [n for n, st in curr_state.items() if st == "R"]

    ax1.clear()
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=i_list, node_color="blue", node_size=15, ax=ax1)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=s_list, node_color="red", node_size=15, ax=ax1)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=r_list, node_color="green", node_size=15, ax=ax1)
    ax1.set_title(f"Network at time {i}")

    ax2.clear()
    ax2.plot(x[:i], iss[:i], color="blue", label="Ignorant")
    ax2.plot(x[:i], sss[:i], color="red", label="Spreader")
    ax2.plot(x[:i], rss[:i], color="green", label="Stifler")
    ax2.set_xlim(0, num_iters)
    ax2.set_ylim(0, 4100)
    ax2.set_title("SIR Evolution Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Population")
    ax2.legend()

    return []

ani = animation.FuncAnimation(fig, combined_animation, fargs = (G, pos, time_arr, iss, sss, rss), frames=num_iters, interval=70, blit=False)


plt.tight_layout()
plt.show()