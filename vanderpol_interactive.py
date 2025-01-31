import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent

mu = 1 #on click - this would be what the person selects on the third plot
def ode(t, xy):
    x,y = xy
    dxdt = mu * (y - ((1/3)*(x**3)-x))
    dydt = - x / mu
    return (dxdt, dydt)

t_span = [0, 100]
initial_conditions = [0.0, 0.0] #just initializing this to be 0 and 0 

#plotting, animating, and making it interactive
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout()

#plotting nullclines and phase plane
(plot_trajectory,) = ax1.plot([], [])
(plot_xnullcline,) = ax1.plot([], [], color="red", label="dx/dt nullcline")
(plot_ynullcline,) = ax1.plot([], [], color = "green", label="dy/dt nullcline")
(plot_fixedpoint,) = ax1.plot([], [], "ko", color = "black")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Phase Plane Analysis")
ax1.legend()

#plotting the x versus t
(plot_time,) = ax2.plot([], [])
ax2.set_title("Time Series")
ax2.set_xlabel("t")
ax2.set_ylabel("x")

def animate(i, xy, x_nullcline, y_nullcline, fsolve_res):
    if xy is None:
        return ()
    
    plot_xnullcline.set_data(x_nullcline[0], x_nullcline[1])
    plot_ynullcline.set_data(y_nullcline[0], y_nullcline[1])
    plot_fixedpoint.set_data([fsolve_res[0]], [fsolve_res[1]])
    plot_trajectory.set_data(xy[0][:i], xy[1][:i])
    plot_time.set_data(t_span[: i + 1], xy[0][: i + 1])
    return (plot_trajectory, plot_xnullcline, plot_ynullcline, plot_fixedpoint, plot_time)

anim = animation.FuncAnimation(fig, animate, interval = 1, blit = False)
anim.event_source.stop()

def mouse_click(event: MouseEvent):
    if event.inaxes == ax1:
        initial_conditions[0] = event.xdata #lists are mutable so this is going to change the initial conditions
        initial_conditions[1] = event.ydata
    else:
        return
    
    solution = solve_ivp(ode, t_span, initial_conditions)
    xvals_sol = solution.y[0]
    yvals_sol = solution.y[1]
    y = solution.y

    #the nullclines:
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
    x_null = np.array([xx_nullcline_pts_sorted, xy_nullcline_pts_sorted])
    #doing the same for the y
    yx_sorted_indices = np.argsort(yx_nullcline_pts)
    yx_nullcline_pts_sorted = yx_nullcline_pts[yx_sorted_indices]
    yy_nullcline_pts_sorted = yy_nullcline_pts[yx_sorted_indices]
    y_null = np.array([yx_nullcline_pts_sorted, yy_nullcline_pts_sorted])

    #identifying the critical point
    #happens when the nullclines "intersect"
    #using fsolve
    #defining dxdt and dydt in a function
    def derivatives(xy):
        x, y = xy
        return [mu * (y - ((1/3)*(x**3)-x)), - x / mu]
    fsolve_res = fsolve(derivatives, x0 = np.array([0,0]))
    print(fsolve_res) #this is the root, which is the critical/fixed point.

    #handling the animation so that it restarts on click
    anim.event_source.stop()
    anim.frame_seq = anim.new_frame_seq()
    anim._args = (y, x_null, y_null, fsolve_res)
    anim.event_source.start()

    #connecting:
    fig.canvas.mpl_connect("button_press_event", mouse_click)

    #show plot
    plt.show()
