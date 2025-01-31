import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.animation as animation

mu = 1 #on click - this would be what the person selects on the third plot
def ode(t, xy):
    x,y = xy
    dxdt = mu * (y - ((1/3)*(x**3)-x))
    dydt = - x / mu
    return (dxdt, dydt)

#phase portrait / trajectories:
initial_conditions = [-2, 3] #on click, this would be where the person presses on the plot. 
t_span = [0, 100]
solution = solve_ivp(ode, t_span, initial_conditions, t_eval=np.linspace(0, 100, 1000)) #this is what needs to be animated
xvals_sol = solution.y[0]
yvals_sol = solution.y[1]
print(f"xvals_sol.shape = {xvals_sol.shape}")
print(f"yvals_sol.shape = {yvals_sol.shape}")
#plt.plot(xvals_sol, yvals_sol, color = "blue", label = "trajectory")
#plt.title("Van der Pol Visualization")
#plt.xlabel("x")
#plt.ylabel("y")

#nullclines:
ylims = np.linspace(-4, 4, 500)
xlims = np.linspace(-4, 4, 500)
xv, yv = np.meshgrid(ylims, xlims)
print(f"xv.shape = {xv.shape}")

#defining the differential equations
dxdt = mu * (yv - ((1/3)*(xv**3)-xv))
dydt = - xv / mu

#finding the nullclines numerically
#mask to see if there is a difference in sign
mask_x = np.diff(np.sign(dxdt)).astype(bool)
print(f"mask_x shape = {mask_x.shape}")
print(np.count_nonzero(mask_x == True))
mask_y = np.diff(np.sign(dydt)).astype(bool)
print(f"mask_y shape = {mask_y.shape}")
print(np.count_nonzero(mask_y == True)) 
#I need to use np.concatenate: need to add a false to the end of each array, 
# because the shape of mask_y and mask_x is not the same as xv and yv
falsearr = np.zeros((xv.shape[1], 1)).astype(bool)
extended_mask_x = np.concatenate((mask_x, falsearr), axis=1)
extended_mask_y = np.concatenate((mask_y, falsearr), axis=1) #these are (100, 100)
#masking:
xx_nullcline_pts = xv[extended_mask_x]
xy_nullcline_pts = yv[extended_mask_x]
print(xx_nullcline_pts.shape)
print(xy_nullcline_pts.shape)
#same for y:
yx_nullcline_pts = xv[extended_mask_y]
yy_nullcline_pts = yv[extended_mask_y]
print(yx_nullcline_pts.shape)
print(yy_nullcline_pts.shape)

#sorting with np.argsort so that the plots become smoother:
xx_sorted_indices = np.argsort(xx_nullcline_pts)
xx_nullcline_pts_sorted = xx_nullcline_pts[xx_sorted_indices]
xy_nullcline_pts_sorted = xy_nullcline_pts[xx_sorted_indices]
#doing the same for the y
yx_sorted_indices = np.argsort(yx_nullcline_pts)
yx_nullcline_pts_sorted = yx_nullcline_pts[yx_sorted_indices]
yy_nullcline_pts_sorted = yy_nullcline_pts[yx_sorted_indices]

#plotting
#plt.plot(xx_nullcline_pts_sorted, xy_nullcline_pts_sorted, color = "red", label = "x_nullcline", ls = ":")
#plt.plot(yx_nullcline_pts_sorted, yy_nullcline_pts_sorted, color = "green", label = "y_nullcine", ls = ":")


#identifying the critical point
#happens when the nullclines "intersect"
#using fsolve
#defining dxdt and dydt in a function
def derivatives(xy):
    x, y = xy
    return [mu * (y - ((1/3)*(x**3)-x)), - x / mu]
fsolve_res = fsolve(derivatives, x0 = np.array([0,0]))
print(fsolve_res) #this is the root, which is the critical/fixed point.

#plt.scatter(fsolve_res[0], fsolve_res[1], color = "black", label = "fixed point")
#plt.legend()
#plt.show()



#animation of trajectory:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

(plot_trajectory,) = ax1.plot([], [])

ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)

def animate(i, x, y): #creating an animation function, which is called for each frame
    #x, y = xy
    plot_trajectory.set_data(x[:i], y[:i])
    return plot_trajectory

ani1 = animation.FuncAnimation(fig, animate, fargs=(solution.y[0], solution.y[1]), interval=70, blit=False)
ax1.plot(xx_nullcline_pts_sorted, xy_nullcline_pts_sorted, color = "red", label = "dx/dt nullcline", ls = ":")
ax1.plot(yx_nullcline_pts_sorted, yy_nullcline_pts_sorted, color = "green", label = "dy/dt nullcine", ls = ":")
ax1.scatter(fsolve_res[0], fsolve_res[1], color = "black", label = "fixed point")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.set_title("Van der Pol Phase Plane")


(plot_time,) = ax1.plot([], [])
#animating a time plot that shows time on the x axis and x on the y axis
def animate_time(i, t, x):
    plot_time.set_data(t[:i], x[:i])
    return plot_time

(plot_time,) = ax2.plot([], [])
ax2.set_xlim(0, 100)
ax2.set_ylim(-4, 4)

ani2 = animation.FuncAnimation(fig, animate_time, fargs=(solution.t, solution.y[0]), interval = 70, blit = False)
ax2.set_ylabel("x")
ax2.set_xlabel("t")
ax2.set_title("Time Series")

plt.tight_layout()
plt.show()