import streamlit as st
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

st.title("Spruce Budworm Population Dynamics")
st.divider()
growth_rate_r = st.slider("Intrinsic Growth Rate (r)", min_value = 0.0, max_value=1.0, step = 0.01)
carrying_capacity_k = st.slider("Carrying Capacity (k)", min_value = 0.0, max_value=10.0, step = 0.01)

r=growth_rate_r
k=carrying_capacity_k
def ode(t, y):
    x = y
    dxdt = r * x * (1 - x/k) - ((x ** 2) / (1 + x ** 2))
    return (dxdt)

#plot for the population
population = np.arange(0, 8, step = 0.1)
ratesofchange = []
for i in population:
    rateofchange = ode(1, i)
    ratesofchange.append(rateofchange)

fig = plt.figure()
plt.plot(population, ratesofchange, color = "black")
zeroline = np.zeros_like(population)
plt.plot(population, zeroline, ls = "--", color = "gray")
plt.grid(alpha = 0.3)
plt.title("Spruce Budworm Rate of Change")
plt.xlabel("Population")
plt.ylabel("Rate of Change")

#now we also want to add the zeros - this is when the rate of change is equal to 0
mask = np.diff(np.sign(ratesofchange)).astype(bool) #we look at where the sign changes - meaning there has been a zero
posneg = np.diff(np.sign(ratesofchange))
zerovals = posneg[posneg != 0]
print(posneg)
print(zerovals)
print(mask)
#print(mask.shape)
appmask = np.append(mask, False) #because the dimensions dont match...
zeroes = population[appmask] #this gives us the critical points in terms of population values
print(zeroes)
print(zeroes.dtype)

#looking at stability:
#unstable:
posindices = np.where(zerovals > 0)[0] #finding where the pts changed positively therefore unstable
print(posindices)
unstablepts = []
for i in posindices:
    unstablepts.append(zeroes[i])
print(f"unstablepts = {unstablepts}")
#stable:
negindices = np.where(zerovals < 0)[0] 
print(negindices)
stablepts = []
for i in negindices:
    stablepts.append(zeroes[i])
print(f"stablepts = {stablepts}")
ymin, ymax = plt.ylim()
plt.vlines(k/10, ymin=ymin, ymax=ymax, label="Initial Population", ls="--", color = "green")
plt.scatter(unstablepts, np.zeros_like(unstablepts), color="red", s=20, label = "Unstable")
plt.scatter(stablepts, np.zeros_like(stablepts), color="blue", s=20, label = "Stable")
plt.legend()

st.pyplot(fig)

st.divider()

# streamlit run budworm.py

#now, we need to create a graph showing the evolution of the budworm population over time
st.markdown("Now, you can observe how the budworm population evolves over time based on the chosen parameters. Keep pressing "
            "evolve to run the simulation and reset if you want to start over. The default initial population is $\\frac{k}{10}$.")
col1, col2 = st.columns(2)
time_for_evolution = col1.slider(label="Time", min_value=0, max_value=100)
evolve_button = col2.button(label="Press to Evolve")
reset_button = col2.button(label="Press to Reset")
 
initial_pop = carrying_capacity_k / 10 #the initial pop is always k/10

#when we press the evolve button, we want to integrate at a time step of time_for_evolution
#so if we start at 0, and then press evolve, we want to see what the population looks like from 0 to time_for_evolution.
#and then, when we press evolve again, we want to go from 0 to  time_for_evolution*2
#when we press reset, we want to start again from 0. 

if "time" not in st.session_state:
    st.session_state["time"] = 0
if "solutions" not in st.session_state:
    st.session_state["solutions"] = [] #solutions
if "last_time" not in st.session_state:
    st.session_state["last_time"] = 0 #last time point


#start with t = 0, initial population k/10
if evolve_button:
    if time_for_evolution == 0:
        st.text("Set time to a value other than 0")
    else:
        start_time = st.session_state["last_time"] #lower bound of t
        end_time = start_time + time_for_evolution
        st.session_state["time"] = np.arange(start_time, end_time + 0.5, step = 0.5)

        t_span_0 = st.session_state["time"][0]
        t_span_1 = st.session_state["time"][-1]
        t_span = [t_span_0, t_span_1] #the bounds we will need to integrate from

        if st.session_state["last_time"] == 0:
            init_conditions = [initial_pop]
        else:
            init_conditions = [st.session_state["solutions"][-1][1][-1]] #extracting the last value of solutions
            print(init_conditions)

        solution = solve_ivp(ode, t_span, init_conditions)
        tsol = solution.t
        ysol = solution.y[0]

        #updating the last time point:
        st.session_state["last_time"] = t_span_1

        #storing solutions for plotting:
        st.session_state["solutions"].append((tsol, ysol))

        #plot:
        fig = plt.figure()
        for tsol, ysol in st.session_state["solutions"]:
            plt.plot(tsol, ysol, color = "black")
        plt.xlabel("Time")
        plt.ylabel("Budworm Population")
        plt.grid()
        plt.title("Spruce Budworm Population Dynamics")
        st.pyplot(fig)

if reset_button:
    st.session_state["solutions"] = []
    st.session_state["last_time"] = 0
    st.text("Press evolve to run the simulation.")
