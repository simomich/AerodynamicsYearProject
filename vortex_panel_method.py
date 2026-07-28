"""
author:                  Simone Micelli
email:        simonemicelli47@gmail.com
date:                        31/03/2026
description:
    Educational implementation of the second-order vortex panel method

Copyright (CC) 2026 Simone Micelli
"""

import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from scipy.interpolate import make_splprep

calculate_airfoil_geometric_properties = True
plot_cm_at_x_ac = False
savefigures = False

if savefigures:
    plt.style.use(["science", "high-vis"])
else:
    plt.style.use(["science", "high-vis", "no-latex"])

plt.rcParams.update({
    # "figure.figsize":                (3.5, 2.5),  # grandezza figures
    "savefig.dpi":                   600,         # risoluzione al salvataggio
    "figure.dpi":                    250,         # risoluzione alla visualizzazione
    "axes.grid":                     True,        # griglia degli axes
    "grid.linestyle":                "--",        # stile della linea
    "grid.alpha":                    0.3,         # alpha color della griglia
    "figure.constrained_layout.use": True,        #
    # "figure.autolayout": True,             #
    "text.latex.preamble": r"""\usepackage[T1]{fontenc}
    \usepackage{XCharter}
    \usepackage[xcharter]{newtxmath}""",
},
)


class VortexObject:

    def __init__(self, num_p: int):
        self.Np = num_p

        # geometry arrays
        self.x = None
        self.y = None
        self.xc = None
        self.yc = None
        self.theta = None
        self.beta = None
        self.S = None

        # influence coefficient matrices
        # A: normal velocity influences (used for zero-penetration boundary condition)
        # B: tangential velocity influences (used for surface velocity computation)
        self.A = np.zeros((num_p + 1, num_p + 1))
        self.B = np.zeros((num_p + 1, num_p + 1))
        self.gamma = None

    def problem_formulation(self):
        """
        Computes the geometric parameters and the influence coefficient matrices
        (A and B) for the linear vortex panel method.
        """
        # define control points (collocation points) at the center of each panel
        self.xc = self.x[0:-1] + 0.5 * np.diff(self.x)
        self.yc = self.y[0:-1] + 0.5 * np.diff(self.y)

        # compute panel orientations and lengths
        # IMPORTANT the curve must be traversed clockwise
        self.theta = np.arctan2(np.diff(self.y), np.diff(self.x))
        self.beta = self.theta + 0.5 * np.pi

        # normal versor centered on each centroid
        self.nx = np.cos(self.beta)
        self.ny = np.sin(self.beta)

        self.S = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2)

        # build the influence matrices
        # outer loop: control point i where boundary conditions are evaluated
        for i in range(self.Np):
            # distance from control point i to all panel start nodes j
            xt = self.xc[i] - self.x[0:-1]
            yt = self.yc[i] - self.y[0:-1]

            # rotate coordinates into the local frame of reference of panel j
            xtp = xt * np.cos(self.theta) + yt * np.sin(self.theta)
            ytp = -xt * np.sin(self.theta) + yt * np.cos(self.theta)

            # distances (r1, r2) and angles (theta1, theta2) from the control point
            # to the edges of panel j in its local frame
            r1 = np.hypot(xtp, ytp)
            r2 = np.hypot((xtp - self.S), ytp)
            theta1 = np.arctan2(ytp, xtp)
            theta2 = np.arctan2(ytp, xtp - self.S)

            f = 1.0 / (2 * np.pi * self.S)

            # A1, A2: tangential induced velocity coefficients in the local frame
            # B1, B2: normal induced velocity coefficients in the local frame
            A1 = f * (ytp * np.log(r2 / r1) + (xtp - self.S) * (theta2 - theta1))
            A2 = -f * (ytp * np.log(r2 / r1) + xtp * (theta2 - theta1))

            B1 = f * (self.S + (xtp - self.S) * np.log(r2 / r1) - ytp * (theta2 - theta1))
            B2 = -f * (self.S + xtp * np.log(r2 / r1) - ytp * (theta2 - theta1))

            # velocity self-induction (when control point i is on panel j)
            A1[i] = 0.5 * (xtp[i] / self.S[i] - 1.0)
            A2[i] = -0.5 * xtp[i] / self.S[i]
            B1[i] = 0.5 / np.pi
            B2[i] = -0.5 / np.pi

            # rotate the induced velocities back to the GLOBAL frame of reference (u, v)
            u1 = A1 * np.cos(self.theta) - B1 * np.sin(self.theta)
            u2 = A2 * np.cos(self.theta) - B2 * np.sin(self.theta)
            v1 = A1 * np.sin(self.theta) + B1 * np.cos(self.theta)
            v2 = A2 * np.sin(self.theta) + B2 * np.cos(self.theta)

            # project global velocities into the LOCAL frame of control point i
            # A: normal velocity influence coefficients (dot product with normal vector)
            self.A[i, 0:self.Np] = u1 * np.cos(self.beta[i]) + v1 * np.sin(self.beta[i])
            self.A[i, 1:self.Np + 1] += u2 * np.cos(self.beta[i]) + v2 * np.sin(self.beta[i])

            # B: tangential velocity influence coefficients (dot product with tangent vector)
            self.B[i, 0:self.Np] = u1 * np.cos(self.theta[i]) + v1 * np.sin(self.theta[i])
            self.B[i, 1:self.Np + 1] += u2 * np.cos(self.theta[i]) + v2 * np.sin(self.theta[i])

        # Kutta's condition
        # enforces that the circulation at the trailing edge is finite, meaning
        # the vortex strength at the first and last nodes must cancel out: gamma_1 + gamma_N = 0
        self.A[self.Np, 0] = 1.0
        self.A[self.Np, self.Np] = 1.0

    def solve(self, V: float, alpha: float) -> np.ndarray:
        """
        Solves the linear system to find the vortex strengths (gamma) and
        computes the tangential velocity (Vs) on the surface.

        :param V: freestream velocity [m/s]
        :param alpha: AOA [rad]
        """
        self.V = V

        # build the RHS vector 'b'
        # b_i represents the freestream velocity normal to panel i
        # the sum of induced normal velocity (A * gamma) and freestream normal
        # velocity (b) must be 0 to satisfy the no-penetration boundary condition
        b = np.zeros(len(self.x))
        b[0:-1] = -V * np.cos(self.beta - alpha)
        b[-1] = 0

        # numerical method for the soluzion of the linear system [A]{gamma} = {b}
        self.gamma = np.linalg.solve(self.A, b)

        # calculate tangential surface velocity (Vs) at each control point
        Vs = np.zeros(self.xc.shape)

        for i in range(len(self.xc)):
            # velocity induced by the vortices
            Vs[i] = np.dot(self.B[i, :], self.gamma)

        # add the freestream tangential velocity component
        Vs += V * np.sin(self.beta - alpha)

        return Vs

    def getCl(self, cref=1.0):
        """
        Computes the 2D Lift Coefficient (Cl) using the Kutta-Joukowski theorem.

        :param cref: Reference chord length used for a-dimensionalization (default: 1.0)
        """
        # calculate the cumulative arc length 's' along the panels
        s = np.zeros(self.x.shape)
        for i, S in enumerate(self.S):
            s[i + 1] = s[i] + S

        # integrate the vortex strength (gamma) over the perimeter (s) to get the Total Circulation (Gamma).
        # use the trapezoidal rule because gamma varies linearly across the panels.
        Gamma = np.trapezoid(self.gamma, s)

        # Kutta-Joukowski theorem: L = rho * V_inf * Gamma
        # Cl = L / (0.5 * rho * V_inf^2 * cref) = 2 * Gamma / (V_inf * cref)
        # Note: The negative sign accounts for the standard circulation sign convention.
        return -Gamma / (0.5 * self.V * cref)

    def getCm(self, vs, xref=0.25, yref=0.0, cref=1.0, cp=None):
        """
        Computes the 2D Moment Coefficient (Cm) by integrating
        the pressure distribution over the panels.

        :param vs: tangential surface velocity at each control point (can be evaluated using .solve(...))
        :param xref: X-coordinate of the moment reference center (default: quarter-chord)
        :param yref: Y-coordinate of the moment reference center (default: 0.0)
        :param cref: Reference chord length used for a-dimensionalization (default: 1.0)
        :param cp: If already calculated we can get cp array as an argument
        :return: Cm (Positive is nose moving upward)
        """

        # calculate the pressure coefficient (Cp) for each panel
        if cp is None:
            cp = 1.0 - (vs / self.V) ** 2

        # calculate the geometric deltas (traversing clockwise)
        dx = np.diff(self.x)
        dy = np.diff(self.y)

        # calculate moment arms from the reference point to each control point
        x_arm = self.xc - xref
        y_arm = self.yc - yref

        # calculate the moment contribution of each panel
        dCm = cp * (x_arm * dx + y_arm * dy)

        # sum the contributions and normalize by the chord squared
        Cm = np.sum(dCm) / (cref ** 2)

        return Cm


class CircleVortex(VortexObject):

    def __init__(self, r0, num_p: int):
        """
        Generates a circular cylinder for potential flow analysis.

        :param r0: Cylinder radius
        :param num_p: Number of panels
        :return: CircleVortex object
        """

        super().__init__(num_p)

        # equispaced points along the circumference
        t = np.linspace(0, 2 * np.pi, num_p + 1)

        self.x = r0 * np.cos(t)
        self.y = r0 * np.sin(-t)  # -theta to easily obtain exit normal

        self.problem_formulation()


class EllipseVortex(VortexObject):

    def __init__(self, a: float, b: float, num_p: int):
        """
        Generates an elliptical profile.

        :param a: Horizontal semi-axis
        :param b: Vertical semi-axis
        :param num_p: Number of panels
        :return: EllipseVortex object
        """

        super().__init__(num_p)

        # equiangular points along the ellipse
        theta = np.linspace(0, 2 * np.pi, num_p + 1)

        self.x = a * np.cos(theta)
        self.y = b * np.sin(-theta)  # -theta to easily obtain exit normal

        self.problem_formulation()


class AirfoilJVortex(VortexObject):

    def __init__(self, r0: float, m: float, delta: float, num_p: int):
        """
        Generates a Joukowski's airfoil using conformal mapping.
        It maps a shifted cylinder in the complex plane to an airfoil shape.

        :param r0: Base circumference radius
        :param m: Translational shift distance toward the second quadrant (thickness/camber control)
        :param delta: Translational shift angle [deg] (controls camber)
        :param num_p: Number of panels
        :return: AirfoilJVortex object
        """

        super().__init__(num_p)

        d = delta  # [deg]
        d = d * (np.pi / 180)

        # calculate the trailing edge mapping parameter (b) and shift coordinates
        self.beta_joukowski = np.arcsin(m * np.sin(d) / r0)
        b = r0 * np.cos(self.beta_joukowski) - m * np.cos(d)
        cx = -1 * m * np.cos(d)  # center X of the base cylinder
        cy = m * np.sin(d)  # center Y of the base cylinder

        # equiangular points along the circumference, starting from the trailing edge point
        theta = np.linspace(-self.beta_joukowski, 2 * np.pi - self.beta_joukowski, num_p + 1)

        # base circumference:
        x_c = cx + r0 * np.cos(theta)
        y_c = cy + r0 * np.sin(theta)
        denom = x_c ** 2 + y_c ** 2

        # apply the Joukowski Transformation separated into real (x) and imaginary (y) components
        # np.flip is used to reverse the array order, ensuring the resulting airfoil
        # is traversed CLOCKWISE (the mapping naturally reverses the original direction).
        self.x = np.flip(x_c * (1 + b ** 2 / denom))
        self.y = np.flip(y_c * (1 - b ** 2 / denom))
        
        # find the index of the leading edge
        le_idx = np.argmin(self.x)
        
        # calculate chord length
        self.chord = np.sqrt((self.x[0] - self.x[le_idx])**2 + (self.y[0] - self.y[le_idx])**2)

        self.problem_formulation()


class NACAExternal(VortexObject):
    """
    Loads an airfoil coordinate file and optionally interpolates it to increase/redistribute the number of panels.

    Args:
        sense: "ClockWise" or "CounterClockWise" choose one of those depending on the point order in the file
        path: file path
        interp: grade of the B-spline (0 for no interpolation)
        new_num_p: number of new points for the interpolated curve
        distr: use a non linear point distribution for better convergence (only cosine-like distribution is
        implemented at the moment)
    """

    def __init__(self, sense: str, path: str, interp: int = 0, new_num_p=None, distr: int = 0):
        # Load external coordinate file
        data = np.loadtxt(path, dtype=np.float64)

        x = data[:, 0]
        y = data[:, 1]

        # ensure the coordinates are ordered Clockwise.
        if sense == "CounterClockWise":
            x = x[::-1]
            y = y[::-1]
        elif sense == "ClockWise":
            pass
        else:
            raise (ValueError(
                "You must specify whether the set of points is ClockWise or CounterClockWise"))

        # optional geometry refinement using B-Splines
        if interp != 0:
            # group the points in a list of shape (2,N)
            points = [x, y]

            ########################################
            # B-spline interpolation
            ########################################
            # make_splprep returns a BSpline instance of scipy.
            # s = 0 forces the interpolation to be exact over the original points
            # (this applies to the final curve we receive if the new parameter array contains the exact values
            # corresponding to the original points, that are contained in the u array; see the return section of
            # the documentation for more info)
            spl, u = make_splprep(points, s=0, k=interp)

            # New points definition
            # By default, make_splprep normalises the curve parameter between 0.0 and 1.0
            if new_num_p is not None:
                if distr == 1:
                    # non-uniform distribution: cosine-like clustering.
                    # this clusters points closer to the leading and trailing edges
                    # where velocity gradients are highest, improving numerical accuracy.
                    lin = np.linspace(0, 1, new_num_p)
                    t_new = lin - np.sin(4 * np.pi * lin) / (4 * np.pi)
                else:
                    # uniform parameter distribution
                    t_new = np.linspace(0, 1, new_num_p)

            else:
                raise (ValueError(
                    "If interp != 0 than the argument new_num_p must be defined"))

            # Evaluate the spline at the newly distributed parameters
            x, y = spl(t_new)

        super().__init__(len(x) - 1)
        self.x = x
        self.y = y
        self.problem_formulation()


def start_sim(vortex_obj: VortexObject, v_inf: float, alpha: float):
    """
    Runs the simulation and plots the geometry and the pressure coefficient (Cp)
    specifically tailored for bluff bodies like cylinders/ellipses.
    """
    alpha *= np.pi / 180
    vs = vortex_obj.solve(v_inf, alpha)

    # compute pressure coefficient using Bernoulli's equation: Cp = 1 - (V/V_inf)^2
    cp = 1 - (vs / v_inf) ** 2

    # plot 1: geometry and control Points
    plt.gca().axis("equal")
    plt.plot(vortex_obj.x, vortex_obj.y)
    plt.scatter(vortex_obj.xc, vortex_obj.yc, c='r', s=0.5)
    plt.show()

    # calculate angular position for the x-axis of the Cp plot
    theta = np.arctan2(vortex_obj.yc, vortex_obj.xc)

    # Sort the indices to prevent a messy zig-zag line in the plot where theta jumps from pi to -pi
    idx = np.argsort(theta)

    # plot 2: pressure coefficient vs angle
    plt.plot(theta[idx], cp[idx], c="black")

    if type(vortex_obj) == CircleVortex and alpha == 0:
        # if simulating a cylinder, overlay the exact analytical solution for comparison.
        # analytical Cp for cylinder: Cp = 1 - 4*sin^2(theta)
        theta_p = np.linspace(-np.pi, np.pi, 1000)
        plt.plot(theta_p, 1 - 4 * (np.sin(theta_p))
                 ** 2, c="orange", alpha=0.7)

    plt.tight_layout()
    plt.show()


def start_sim_airfoil(vortex_obj: VortexObject, v_inf: float, a: float):
    """
    Runs the simulation and plots the geometry and Cp tailored for airfoils.
    Plots Cp against the x-coordinate instead of the angle.
    """
    a *= np.pi / 180
    vs = vortex_obj.solve(v_inf, a)

    cp = 1 - (vs / v_inf) ** 2

    # plot airfoil geometry and control points
    plt.gca().axis("equal")
    plt.plot(vortex_obj.x, vortex_obj.y)
    plt.title("Airfoil geometry")
    plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
    plt.ylabel(r"$\frac{y}{c}\;\mathrm{[-]}$")
    # plt.scatter(vortex_obj.xc, vortex_obj.yc, c='r', s=0.5)
    plt.show()

    # plot pressure distribution over the chord length
    plt.plot(vortex_obj.xc, cp, c="black")
    plt.title("Pressure coefficient")
    plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
    plt.ylabel(r"$C_p\;\mathrm{[-]}$")
    plt.show()


def validate_code(r0, m, delta, num_p):
    """
    Code validation, it is valid if the solver finds a Lift value error that is less than 2% for the Joukowski'airfoil 
    specified below with respect to the theoretical Lift value formula.
    
    :param r0: Base circumference radius
    :param m: Translational shift distance toward the second quadrant (thickness/camber control)
    :param delta: Translational shift angle [deg] (controls camber)
    :param num_p: Number of panels
    :return: AirfoilJVortex object
    """
    
    VINFS = np.array([x for x in range(10, 30+1, 10)], dtype=np.float64)
    ALPHAS = np.array([x for x in range(-1, 4+1, 1)], dtype=np.float64) * np.pi / 180.0
    
    airfoil_j = AirfoilJVortex(r0, m, delta, num_p)
    
    print(f"""{"="*74}
Validation based on the Joukowski's airfoil with the following parameters:
r0    = {r0}
m     = {m}
delta = {delta}
num_p = {num_p}
{"="*74}""")
    
    for i in range(len(VINFS)):
        
        for j in range(len(ALPHAS)):
            
            vs = airfoil_j.solve(VINFS[i], ALPHAS[j])
            
            Cl = airfoil_j.getCl(cref=airfoil_j.chord)
            rho_calc = 1.225
            L_calc = Cl * 0.5 * rho_calc * (VINFS[i] ** 2) * airfoil_j.chord
            
            L_theory = rho_calc * (VINFS[i] ** 2) * 4 * np.pi * r0 * np.sin(ALPHAS[j] + airfoil_j.beta_joukowski)
            
            print(f"\033[0m{VINFS[i]} m/s  ~  {ALPHAS[j] * 180 / np.pi} deg:")
            
            # dealing with no-lift generation
            if L_theory == 0:
                print("\033[1;33mTHE AIRFOIL IN THEORY DOESN'T GENERATE ANY LIFT\033[0m")
                print(f"\033[1;33mCalculated Lift value: {L_calc:0.001E} N\033[0m")
            else:
                deviation = (L_theory - L_calc) / L_theory
                
                if deviation <= 0.02:
                    print("\033[1;32mValidated\033[0m")
                else:
                    print("\033[31mNOT VALIDATED\033[0m")
                    print(f"\033[31mDeviation with the theoretical value: \033[31m{deviation * 100:0.001f} %\033[0m")
            
    #         print(f"""{VINFS[i]} m/s  ~  {ALPHAS[j] * 180 / np.pi} deg:
    # L_calc   = {L_calc}
    # L_theory = {rho_calc * (VINFS[i] ** 2) * 4 * np.pi * r0 * np.sin(ALPHAS[j] + airfoil_j.beta_joukowski)}""")


if __name__ == "__main__":
    # circle = CircleVortex(1.0, 50)
    # vinf = 1.0
    # alpha = 10
    # start_sim(circle, vinf, alpha)

    # ellipse = EllipseVortex(2.0, 1.0, 50)
    # vinf = 1.0
    # alpha = 10
    # start_sim(ellipse, vinf, alpha)
    #
    # airfoil_j = AirfoilJVortex(1.0, 0.2, 30, 100)
    # vinf = 1.0
    # alpha = 0
    # start_sim_airfoil(airfoil_j, vinf, alpha)
    
    
    # code validation block using Joukowski's Airfoil
    if False:
        validate_code(1.0, 0.2, 30, 3000)
        validate_code(1.0, 0.07, 30, 3000)
        validate_code(1.0, 0.16, 0, 4000)
    
    
    # load the desired profile from a file, applying a (cubic: interp=3) spline interpolation
    # to smooth it and increase panel density
    NACA = NACAExternal("CounterClockWise", "NACA Airfoils/2026_sup.dat",
                        interp=3, new_num_p=500, distr=1,
                        )
    
    print(f"""{"="*80}
{"Airfoil characteristics | Second order vortex panel method":^80}
{"="*80}""")

    if calculate_airfoil_geometric_properties:
        #####################################################################
        # Maximum thickness and camber value and locations + Camber line plot
        #####################################################################

        # may not be the faster or precise implementation but it seems indeed straightforward to me

        # find the leading edge index (minimum X coordinate)
        le_idx = np.argmin(NACA.x)

        # slice the arrays based on the Clockwise traversal
        # note: they both share the leading edge point
        # starting from TE (index 0) to LE (le_idx) -> lower surface
        pressure_side = (NACA.x[:le_idx + 1], NACA.y[:le_idx + 1])

        # from LE (le_idx) back to TE (end of array) -> upper surface
        suction_side = (NACA.x[le_idx:], NACA.y[le_idx:])

        xs_arr = pressure_side[0]
        ys_arr = pressure_side[1]

        x_t_max = 0
        y_pr_t_max = 0
        y_su_t_max = 0
        t_max = 0
        x_camber_max = 0
        camber_max = 0
        camber_line = []
        for idx, pr_x in enumerate(xs_arr):
            pr_y = ys_arr[idx]

            su_idx = np.abs(suction_side[0] - pr_x).argmin()

            su_x = suction_side[0][su_idx]
            su_y = suction_side[1][su_idx]

            # print(pr_x, pr_y)
            # print(su_x, su_y)

            t = su_y - pr_y
            if t_max < t:
                t_max = t
                x_t_max = pr_x
                y_pr_t_max = pr_y
                y_su_t_max = su_y

            c = 0.5 * (su_y + pr_y)
            camber_line.append(c)
            if camber_max < c:
                camber_max = c
                x_camber_max = pr_x
        
        print(f"""|{"Geometric properties":^78}|
{"|t_max = ":<20}{t_max:>18.4f} | {"x_t_max = ":<20}{x_t_max:>18.4f}|
{"|camber_max = ":<20}{camber_max:>18.4f} | {"x_camber_max = ":<20}{x_camber_max:>18.4f}|""")

    # plot airfoil geometry and control points (even camber line if properties have been calculated)
    plt.gca().axis("equal")
    plt.plot(NACA.x, NACA.y)
    if calculate_airfoil_geometric_properties:
        # label of the camber line
        plt.plot(xs_arr, camber_line, linewidth=0.5, label="camber line")
        plt.legend(fontsize=6)

        # t_max segment
        plt.plot((x_t_max, x_t_max), (y_pr_t_max, y_su_t_max), linewidth=0.5, c='black')
        # text annotation of t_max
        plt.annotate(
            r"$t_{max}$",                # Text
            xy=(x_t_max, y_su_t_max),    # The exact point to annotate
            xytext=(10.5, -8.5),           # Offset the text by 0 points horizontally and 15 points vertically
            textcoords="offset points",  # Tell matplotlib that xytext is in points, not data coordinates
            ha="center",                 # Horizontally center the text over the point
            va="bottom",                 # Align the bottom of the text with the offset location
            fontsize=8,
            color="black"                # Text color
        )

    plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
    plt.ylabel(r"$\frac{y}{c}\;\mathrm{[-]}$")
    # plt.scatter(NACA.xc, NACA.yc, c='r', s=0.5)
    if not savefigures:
        plt.title("Airfoil geometry")
    else:
        if calculate_airfoil_geometric_properties:
            plt.savefig("saved_plots/airfoil_geometry_with_properties.pdf", format="pdf")
        else:
            plt.savefig("saved_plots/airfoil_geometry.pdf", format="pdf")
    
    # plt.savefig("saved_plots/interpolated_airfoil_geometry.pdf", format="pdf")
    # plt.savefig("saved_plots/non_interpolated_airfoil_geometry.pdf", format="pdf")
    plt.show()

    # NACA = NACAExternal("ClockWise", "NACA Airfoils/v23010.dat.txt",
    #                     interp = 3, new_num_p = 500, distr = 0,
    #                     )

    vinf = 40.0
    # alpha = 10

    # # run the single simulation
    # start_sim_airfoil(NACA, vinf, alpha)
    # print(f"NACA con alpha = {alpha} → Cl = {NACA.getCl():.5f}")

    # Cp & Lift & Moment Curves Calculation
    # sweep through angles of attack (alpha)
    Cl_list = []
    Cm_list = []
    alphas = np.arange(-10, 10.1, 0.25, dtype=np.float64)
    alphas_in_legend = [0, 1, 2, 3, 4]

    # indexes' array of alphas corresponding to alphas_in_legend
    indices_to_plot = np.where(np.isin(alphas, alphas_in_legend))[0]

    cps = []
    for i, alpha in enumerate(alphas):
        alpha_deg = alpha
        alpha *= np.pi / 180
        vs = NACA.solve(vinf, alpha)

        # pressure coefficient
        cp = 1 - (vs / vinf) ** 2

        # plot pressure coefficient distribution over the airfoil
        if i in indices_to_plot:
            # single alpha fixed cp plot showing the airfoil
            plt.gca().axis("scaled")
            plt.plot(NACA.x, NACA.y, c="blue", linewidth=1.5)
            plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
            plt.ylabel(r"$\frac{y}{c}\;\mathrm{[-]}$")

            # Cp scale factor (plot by drawing an arrow of magnitude equal to scale_factor * cp)
            scale_factor = 0.5

            # a step of 3 means it will plot an arrow every three elements in the array.
            step = 6

            pos_mask = cp > 0
            neg_mask = cp <= 0

            # positive Cp (red): pressure pushing into the airfoil
            plt.quiver(NACA.xc[pos_mask][::step], NACA.yc[pos_mask][::step],
                       -NACA.nx[pos_mask][::step] * cp[pos_mask][::step],
                       -NACA.ny[pos_mask][::step] * cp[pos_mask][::step],
                       color="red", scale_units="xy", angles="xy", scale=1/scale_factor, width=0.003, pivot="tip")

            # negative Cp (green): suction pulling out of the airfoil
            plt.quiver(NACA.xc[neg_mask][::step], NACA.yc[neg_mask][::step],
                       NACA.nx[neg_mask][::step] *
                       np.abs(cp[neg_mask][::step]),
                       NACA.ny[neg_mask][::step] *
                       np.abs(cp[neg_mask][::step]),
                       color="green", scale_units="xy", angles="xy", scale=1/scale_factor, width=0.003)

            # cp distribution line
            plt.plot(NACA.xc + scale_factor * NACA.nx * np.abs(cp),
                     NACA.yc + scale_factor * NACA.ny * np.abs(cp),
                     c="black", linestyle='-', linewidth=0.5, alpha=0.6)

            plt.xlim(-0.6, 1.2)
            plt.ylim(-0.7, 0.7)

            if not savefigures:
                plt.title(fr"Cp distr over the airfoil $\alpha={
                          alpha_deg}\;\mathrm{{deg}}$")
            else:
                plt.savefig(
                    f"saved_plots/cp_distr_fixed_aoa_{alpha_deg}.pdf", format="pdf")

            plt.show()

            # save the cp array in order to plot it in the same graph
            cps.append(cp)
        # show is located outside this for loop in order to plot every curve in one graph

        Cl_list.append(NACA.getCl())
        # here we take advantage of the cp calculated above
        if plot_cm_at_x_ac:
            # Cm vs alpha at the aerodynamic center (calculated before)
            Cm_list.append(NACA.getCm(vs, cp=cp, xref=0.26247845))
        else:
            Cm_list.append(NACA.getCm(vs, cp=cp))

    if not plot_cm_at_x_ac:
        ##################################################
        # aerodynamic center calculation
        ##################################################
        # from lecture notes we derived the following expr
        # x_ac = (0.25 * dCl_dalpha - dCm_c_4_dalpha) / dCl_dalpha
        # so by retrieving the Cl_list and Cm_list's linear (deg=1) regression angular coefficients,
        # we solved the position of x_ac
        # See the Cm_ac vs alpha plot commented above
        dCl_dalpha, Cl0 = np.polyfit(alphas * np.pi/180, Cl_list, 1)
        dCm_c_4_dalpha, Cm0 = np.polyfit(alphas * np.pi/180, Cm_list, 1)

        x_ac = (0.25 * dCl_dalpha - dCm_c_4_dalpha) / dCl_dalpha
        
        # find the alpha L=0
        alphaL0 = -Cl0 / dCl_dalpha * 180/np.pi
        
        # find Cm_x_ac
        vs_ = NACA.solve(vinf, 0)
        cp_ = 1 - (vs_ / vinf) ** 2
        Cm_x_ac = NACA.getCm(vs_, cp=cp_, xref=0.26247845)
        
        
        print(f"""|{" ":^78}|
|{"Aerodynamic properties":^78}|
{"|Cl0 = ":<20}{Cl0:>18.4f} | {"dCl_dalpha = ":<20}{dCl_dalpha:>18.3f}|
{"|alpha_L0 = ":<20}{alphaL0:>14.2f} deg | {" ":<20}{" ":>18}|
{"|Cm0 = ":<20}{Cm0:>18.3e} | {"dCm_c_4_dalpha = ":<20}{dCm_c_4_dalpha:>18.3e}|
{"|x_ac = ":<20}{x_ac:>18.4f} | {"Cm_x_ac = ":<20}{Cm_x_ac:>18.3e}|
{"="*80}""")


    for cp in cps:
        plt.plot(NACA.xc, cp)
    plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
    plt.ylabel(r"$C_p\;\mathrm{[-]}$")
    plt.xlim(-0.075, 1.075)
    plt.ylim(-1.8, 1.2)
    plt.legend([rf"${a} \; \mathrm{{deg}}$" for a in alphas_in_legend], title=r"$\alpha$", fontsize=6,
               loc="upper center", bbox_to_anchor=(0.85, 1.))
    plt.gca().invert_yaxis()

    if not savefigures:
        plt.title("Pressure coefficient")
    else:
        plt.savefig("saved_plots/cp_distribution_varing_aoa.pdf", format="pdf")
    plt.show()

    # due to the fact that our method is based on the potential flow assumption we don't get any value of Cd
    # (D'Alambert's paradox) for just the airfoil
    dummy_cds = np.full_like(alphas, np.nan)
    np.savetxt("airfoilsdat/2026_performance.dat",
               np.transpose([alphas, Cl_list, dummy_cds, dummy_cds, Cm_list]))

    plt.plot(alphas, Cl_list, c="blue")
    plt.xlabel(r"$\alpha\;\mathrm{[deg]}$")
    plt.ylabel(r"$C_l\;\mathrm{[-]}$")
    if not savefigures:
        plt.title("Airfoil characteristics")
    else:
        plt.savefig("saved_plots/airfoil_Cl_vs_aoa.pdf", format="pdf")
    plt.show()

    plt.plot(alphas, Cm_list, c="green")
    plt.xlabel(r"$\alpha\;\mathrm{[deg]}$")
    plt.ticklabel_format(style="sci", axis='y', scilimits=(0, 0))
    plt.ylabel(r"$C_m\;\mathrm{[-]}$")
    if not savefigures:
        plt.title("Pitching moment curve")
    else:
        plt.savefig("saved_plots/airfoil_Cm_vs_aoa.pdf", format="pdf")
    plt.show()
