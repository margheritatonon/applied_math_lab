import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


st.title("Interactive SIR Model")
st.markdown("The system of ODEs representing the SIR model are:")
st.latex(r"\frac{dS}{dt} = -\beta \cdot S \cdot I")
st.latex(r"\frac{dI}{dt} = \beta \cdot S \cdot I - \gamma \cdot I")
st.latex(r"\frac{dR}{dt} = \gamma \cdot I")
st.markdown("Where $\\beta$ represents the transmission rate, $\gamma$ represents the recovery rate, $S$ the susceptible population, $I$ the infected population, and $R$ the recovered population.")

st.divider()

col1, col2 = st.columns(2)
transmission_rate_beta = col1.slider("Transmission Rate", min_value = 0.0, max_value = 1.0, step=0.01)
recovery_rate_gamma = col1.slider("Recovery Rate", min_value = 0.0, max_value = 1.0, step=0.01)
s0 = col2.slider("Initial Susceptible Population", min_value = 0.0, max_value = 1.0, step=0.01)
i0 = col2.slider("Initial Infected Population", min_value = 0.0, max_value = 1.0, step=0.01)
r0 = col2.slider("Initial Recovered Population", min_value = 0.0, max_value = 1.0, step=0.01)

def ode(t, y, beta=transmission_rate_beta, gamma=recovery_rate_gamma): #defining the diffeqs
    s, i, r = y #need to do this bc need to unpack the variable so that scipy works
    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    drdt = gamma * i
    return (dsdt, didt, drdt)


t_span = [0, 50] #integrating from day 0 to day 50
init_conditions = [s0, i0, r0] #same order as the ode
solution = solve_ivp(ode, t_span, init_conditions) #u need to go back into the ode and put t at first, bc scipy needs the t even tho not used
print("\n-")
tsol = solution.t
ysol = solution.y
print(tsol)
print(ysol)

#plot:
susceptibles = ysol[0]
infected = ysol[1]
recovered = ysol[2]
fig = plt.figure()
plt.plot(tsol, susceptibles, color = "blue", label="Susceptible")
plt.plot(tsol, infected, color = "pink", label="Infected")
plt.plot(tsol, recovered, color = "purple", label="Recovered")
plt.xlabel("Days")
plt.ylabel("Percent of Population")
plt.title("SIR Model")
plt.grid()
plt.legend()

st.pyplot(fig)

button_val = st.button("See values at t = 50", type = "secondary", icon = "👓")
if button_val:
    st.text(f"Susceptible population percent is {susceptibles[-1] * 100}")
    st.text(f"Infected population percent is {infected[-1] * 100}")
    st.text(f"Recovered population percent is {recovered[-1] * 100}")

# streamlit run sirmodelstreamlit.py
