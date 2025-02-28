import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, fftfreq

#defining all of the parameters
M = 1
K = 1
F = 25
beta = 0.01
C = 0.01 #2*1.65 * beta #this is zeta
N = 200 #the number of pedestrians
dt = 0.01

sigma = 0.086 #see eckhardt paper
omegas = np.random.normal(1, sigma, N) #sampling N omegas from a normal distribution

thetas_0 = np.random.uniform(0, 2*np.pi, N) #initializing the thetas
x = np.zeros_like(thetas_0)
v = np.zeros_like(thetas_0)

def bridge_odes(t, init_condition, M = M, C = C, K = K, F = F, N = N, omegas = omegas, beta = beta):
    """
    Defines the system of ODEs and returns xdot, vdot, and thetasdot.
    """
    x = init_condition[:N]
    v = init_condition[N:2*N]
    thetas = init_condition[2*N:]

    xdot = v
    vdot = (1/M) * (-C*v - K*x + (1/N)*np.sum(F*np.cos(thetas)))
    thetasdot = omegas - beta * vdot * np.cos(thetas)
    return np.concatenate([xdot, vdot, thetasdot])


#function that calculates the order parameter r
def find_r(theta):
    """
    Calculates the order parameter r based on array of phase angles theta.
    """
    return np.abs(np.sum(np.exp(1j * theta)) / len(theta))

r_versus_t = False
fast_fourier = False

if __name__ == "__main__":
    #evolving the system over time
    initial_condition = np.concatenate([x, v, thetas_0])
    t_final = 160
    t_eval = np.arange(0, t_final, dt)
    print(t_eval.shape)
    sol = solve_ivp(bridge_odes, (0, t_final), initial_condition, t_eval=t_eval)
    xs_sol = sol.y[:N, :]
    vs_sol = sol.y[N:2*N, :]
    thetas_sol = sol.y[2*N:, :]
    print(thetas_sol.shape)
    
    if r_versus_t == True:
        rs = []
        for theta in thetas_sol.T:
            r = find_r(theta)
            rs.append(r)
        print(len(rs))
        print(sol.t.shape)
        plt.plot(sol.t, np.array(rs))
        plt.xlabel('Time (t)')
        plt.ylabel('Order Parameter r')
        plt.title('Order Parameter r vs Time')
        plt.show()

    plt.plot(sol.t, np.mean(xs_sol, axis=0), color='blue')  # Plot mean displacement of bridge
    plt.xlabel("Time (t)", size = 18)
    plt.ylabel("Bridge Displacement x(t)", size = 18)
    plt.title(f"F = {F}", size=30)
    plt.show()

    if fast_fourier == True:
        y_values = np.mean(xs_sol, axis=0)  # Bridge displacement
        N_points = len(y_values)
        freqs = fftfreq(N_points, dt)  # Compute frequencies
        fft_values = np.abs(fft(y_values))  # Compute FFT

        plt.figure(figsize=(8, 4))
        plt.plot(freqs[:N_points // 50], fft_values[:N_points // 50])
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title(f"Fourier Spectrum of Bridge Motion (M={M})")
        plt.show()


