from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def gierer_meinhardt_pde(
    t: float,
    uv: np.ndarray,
    gamma: float = 1,
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30,
    dx: float = 1,
) -> np.ndarray:
    """PDEs for the Gierer-Meinhardt model in 2D

    Parameters
    ----------
    t : float
        Time variable (not used in the function)
    uv : np.ndarray
        Array containing the concentrations of u and v
    gamma : float
        Reaction rate parameter, by default 1
    a : float
        Reaction rate parameter, by default 0.40
    b : float
        Reaction rate parameter, by default 1.00
    d : float
        Diffusion rate parameter of v, by default 30
    dx : float
        Spatial step size, by default 1

    Returns
    -------
    np.ndarray
        Array containing the time derivatives of u and v
    """
    # Compute the Laplacian via finite differences
    lap = -4 * uv

    # Immediate neighbors (up, down, left, right)
    lap += np.roll(uv, shift=1, axis=1)  # up
    lap += np.roll(uv, shift=-1, axis=1)  # down
    lap += np.roll(uv, shift=1, axis=2)  # left
    lap += np.roll(uv, shift=-1, axis=2)  # right
    lap /= dx**2

    # Extract the matrices for u and v
    u, v = uv
    lu, lv = lap

    # Compute the ODEs
    f = a - b * u + u**2 / (v)
    g = u**2 - v
    du_dt = lu + gamma * f  # D1 = 1
    dv_dt = d * lv + gamma * g  # D2= d
    return np.array([du_dt, dv_dt])


def giere_meinhardt_jacobian(
    a: float = 0.4, b: float = 1.00
) -> Tuple[float, float, float, float]:
    """Compute the Jacobian of the Gierer-Meinhardt model."""
    # Steps to compute the Jacobian:
    # Find the fixed point of the system
    # Compute the derivatives of the system at the fixed point
    fu = 2 * b / (a + 1) - b
    fv = -((b / (a + 1)) ** 2)
    gu = 2 * (a + 1) / b
    gv = -1.0
    return fu, fv, gu, gv


def is_turing_instability(a: float = 0.40, b: float = 1.00, d: float = 30) -> bool:
    """Check if the conditions for Turing instability are met."""
    # Evaluate the Jacobian at the fixed point
    fu, fv, gu, gv = giere_meinhardt_jacobian(a, b)
    # Compute the determinant of the Jacobian
    nabla = fu * gv - fv * gu
    # Check the conditions
    cond1 = (fu + gv) < 0  # Trace of the Jacobian
    cond2 = nabla > 0  # Determinant of the Jacobian
    cond3 = (gv + d * fu) > (2 * np.sqrt(d) * np.sqrt(nabla))
    return cond1 & cond2 & cond3


def find_unstable_spatial_modes(
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30.0,
    length_x: float = 20.0,
    length_y: float = 50.0,
    num_modes: int = 10,
    boundary_conditions: str = "neumann",
) -> np.ndarray:
    """
    Find the leading spatial modes (wavenumbers) from linear stability analysis.

    Parameters
    ----------
    a : float
        Reaction parameter a.
    b : float
        Reaction parameter b.
    d : float
        Diffusion coefficient for v (D2).
    length_x : float
        L1 domain length.
    length_y : float
        L2 domain length.
    num_modes : int
        Number of wavenumbers (modes) to sample in [0, num_modes].
    boundary_conditions : str
        Boundary conditions to apply. Either 'neumann' or 'periodic'.

    Returns
    -------
    np.ndarray
        Array with the indices of the unstable modes, from largest to smallest.
    """
    # Evaluate the Jacobian
    fu, fv, gu, gv = giere_meinhardt_jacobian(a, b)
    jac = np.array([[fu, fv], [gu, gv]])

    # For Neumann BC on [0,L], modes k_n = (n*pi)/L
    # We will check n=0,...,num_modes-1
    n_values = np.arange(1, num_modes)
    max_eigs = np.zeros((num_modes, num_modes))

    for x in n_values:
        for y in n_values:
            if boundary_conditions == "neumann":
                # For a 1D domain of length L with Neumann boundaries,
                # possible modes are k = n*pi/L, n = 0,1,2,...
                lambda_x = (x * np.pi / length_x) ** 2
                lambda_y = (y * np.pi / length_y) ** 2
            elif boundary_conditions == "periodic":
                lambda_x = ((x + 1) * np.pi / length_x) ** 2
                lambda_y = ((y + 1) * np.pi / length_y) ** 2
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )
            # Compute the eigenvalues of the Jacobian matrix
            a_n = jac - (lambda_x + lambda_y) * np.diag([1, d])
            sigma1, sigma2 = np.linalg.eigvals(a_n)
            # Discard complex part
            sigma1, sigma2 = sigma1.real, sigma2.real
            max_eigs[x, y] = max(sigma1, sigma2)

    # Sort indices by the eigenvalues
    idx, idy = np.unravel_index(np.argsort(max_eigs, axis=None), max_eigs.shape)
    # Sort from largest to smallest and filter out the non-positive eigenvalues
    num_positives = (max_eigs > 0).sum()
    idx, idy = idx[-1:-num_positives:-1], idy[-1:-num_positives:-1]
    # List the indices into list of tuples
    unstable_modes = [(i, j) for i, j in zip(idx, idy)]
    return unstable_modes


def animate_simulation(
    gamma: float = 1,
    b: float = 1.00,
    dx: float = 1.0,
    dt: float = 0.001,
    anim_speed: int = 100,
    length_x: int = 20,
    length_y: int = 50,
    seed: int = 0,
    boundary_conditions: str = "neumann",
):
    """Animate the simulation of the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    dt : float, optional
        Time step size, by default 0.001
    dx : float, optional
        Spatial step size, by default 0.5
    anim_speed : int, optional
        Number of iterations per frame, by default 10
    length_x : float
        L1 domain length, by default 20
    length_y : float
        L2 domain length, by default 50
    seed : int, optional
        Random seed for reproducibility, by default 0
    boundary_conditions : str, optional
        Boundary conditions to apply, by default "neumann"
    """
    # Initialize some variables - Can be modified by the user
    a = 0.40
    d = 20

    # Fix the random seed for reproducibility
    np.random.seed(seed)

    # Compute the number of points in the 2D domain
    lenx = int(length_x / dx)
    leny = int(length_y / dx)

    # Initialize the u and v fields
    uv = np.ones((2, lenx, leny))
    # Add 1% amplitude additive noise, to break the symmetry
    uv += uv * np.random.randn(2, lenx, leny) / 100

    # ------------------------------------------------------------------------ #
    # INITIALIZE PLOT
    # ------------------------------------------------------------------------ #

    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax_ad: Axes = axs[0]  # a vs d
    ax_uv: Axes = axs[1]  # U vs V

    # ------------------------------------------------------------------------ #
    # A-D PLANE
    # ------------------------------------------------------------------------ #

    # In this plane, we will plot the Turing instability conditions
    # The function we designed can work with arrays, so we will create a
    # meshgrid to compute the Turing instability in the entire plane
    arr_a = np.linspace(0, 1, 1000)
    arr_d = np.linspace(0, 100, 1000)
    mesh_a, mesh_d = np.meshgrid(arr_a, arr_d)
    mask_turing = is_turing_instability(mesh_a, b, mesh_d)

    # The following matplotlib function will plot the Turing instability region
    # We use a contour plot to show the region where the conditions are met
    ax_ad.contourf(mesh_a, mesh_d, mask_turing, cmap="coolwarm", alpha=0.5)

    # We also plot a point, that can be moved by the user
    (plot_adpoint,) = ax_ad.plot([a], [d], color="black", marker="o")

    ax_ad.grid(True)
    ax_ad.set_xlabel("a")
    ax_ad.set_ylabel("d")
    ax_ad.set_title("Turing Space")

    # ------------------------------------------------------------------------ #
    # U-V PLANE
    # ------------------------------------------------------------------------ #

    # Dynamic elements: image and text
    im = ax_uv.imshow(
        uv[1],
        interpolation="bilinear",
        # vmin=0,
        # vmax=10,
        origin="lower",
        extent=[0, length_y, 0, length_x],
    )
    # Initialize text objects to display the leading spatial modes
    plot_text = ax_uv.text(
        0.02 * length_x,
        0.39 * length_y,
        "No Turing's instability",
        fontsize=12,
        verticalalignment="top",
    )
    plot_text2 = ax_uv.text(
        0.02 * length_x,
        0.34 * length_y,
        "Click to change initial conditions",
        fontsize=12,
        verticalalignment="top",
    )

    # Static elements: Plot limits, title and labels
    ax_uv.set_xlabel("y")
    ax_uv.set_ylabel("x")
    ax_uv.set_title("Gierer-Meinhardt Model (2D)")

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # This function will be called at each frame of the animation, updating the line objects

    def animate(frame: int, unstable_modes: list):
        # Access the variables from the outer scope
        nonlocal a, d, uv

        # Iterate the simulation as many times as the animation speed
        # We use the Euler's method to integrate the ODEs
        # We cannot use solve_ivp because we must impose the boundary conditions
        # at each iteration
        for _ in range(anim_speed):
            dudt = gierer_meinhardt_pde(0, uv, gamma=gamma, a=a, b=b, d=d, dx=dx)
            uv = uv + dudt * dt
            # The simulation may explode if the time step is too large
            # When this happens, we raise an error and stop the simulation
            if np.isnan(uv).any():
                raise ValueError("Simulation exploded. Reduce dt or increase dx.")
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[:, 0, :] = uv[:, 1, :]
                uv[:, -1, :] = uv[:, -2, :]
                uv[:, :, 0] = uv[:, :, 1]
                uv[:, :, -1] = uv[:, :, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )

        # Update the displayed image
        im.set_array(uv[1])
        # Redefine the color limits. We make sure that the maximum value is at least
        # 0.1 to avoid noise in the image
        im.set_clim(vmin=uv[1].min(), vmax=uv[1].max() + 0.1)
        # Update the point in the a-d plane
        plot_adpoint.set_data([a], [d])
        if len(unstable_modes) == 0:
            plot_text.set_text("No Turing's instability")
            plot_text2.set_text("")
        else:
            plot_text.set_text(f"Leading spatial mode: {unstable_modes[0]}")
            ls_modes = ", ".join(map(str, unstable_modes[1:4]))
            plot_text2.set_text(f"Unstable modes: {ls_modes}")

        # The function must return an iterable with all the artists that have changed
        return [im, plot_adpoint, plot_text, plot_text2]

    ani = animation.FuncAnimation(fig, animate, fargs=([],), interval=1, blit=True)

    # ------------------------------------------------------------------------ #
    # INTERACTION
    # ------------------------------------------------------------------------ #

    # We define a function that will be called when the user clicks on the plot
    # It will update the initial conditions and restart the animation

    def update_simulation(event: MouseEvent):
        # Access the a and d variables from the outer scope and modify them
        nonlocal a, d, uv

        # The click only works if it is inside the phase plane or stability diagram
        if event.inaxes == ax_ad:
            a = event.xdata
            d = event.ydata
        else:
            return

        # Reset to a constant line
        uv = np.ones((2, lenx, leny)) * uv.mean()
        # Add 1% amplitude additive noise, to break the symmetry
        uv += uv * np.random.randn(lenx, leny) / 100

        unstable_modes = find_unstable_spatial_modes(
            a=a,
            b=b,
            d=d,
            length_x=length_x,
            length_y=length_y,
            boundary_conditions=boundary_conditions,
        )

        # Stop the current animation, reset the frame sequence, and start a new animation
        ani.event_source.stop()
        ani.frame_seq = ani.new_frame_seq()
        ani._args = (unstable_modes,)
        ani.event_source.start()

    # Connect the click event to the update function
    fig.canvas.mpl_connect("button_press_event", update_simulation)

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    animate_simulation()