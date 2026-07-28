"""
author:                  Simone Micelli
email:        simonemicelli47@gmail.com
date:                        31/03/2026
description:
    Educational implementation of the classical thin airfoil theory
    ! Only unit chord is now implemented

Copyright (CC) 2026 Simone Micelli
"""

import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from scipy.interpolate import make_splprep
from scipy.integrate import simpson

savefigures = False
showplots = False if savefigures != True else True

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

class ThinAirfoil():
    def __init__(self, x_camber, z_camber):
        # unit chord
        self.c = 1
        self.x_camber = x_camber[::-1]
        self.z_camber = z_camber[::-1]
        self.x_camber[0] = 0.0
        self.z_camber[0] = 0.0
        
        # integration variables
        xi = self.x_camber
        theta = np.concatenate(([0.0], np.array(np.acos(1 - 2 / self.c * xi[1:-1])), [np.pi]))
        
        # numerical derivative of z with respect to x
        dz_dx = np.gradient(self.z_camber, self.x_camber)
        
        # find alpha - A0 ...
        self.alpha_min_A0 = simpson(dz_dx, theta) / np.pi
        
        # ... A1 ...
        self.A1 = 2 / np.pi * simpson(dz_dx * np.cos(theta), theta)
        
        # ... and A2
        self.A2 = 2 / np.pi * simpson(dz_dx * np.cos(2 * theta), theta)
        
    def solve(self, V, alpha):
        """
        Computes 
        
        :param V: [m/s] freestream velocity module
        :param alpha: [rad] freestream velocity's angle of attack
        :return: 
        """
        A0 = alpha - self.alpha_min_A0
        
        Gamma = self.c * V * (np.pi * A0 + np.pi / 2 * self.A1)
        
        Cl = np.pi * (2 * A0 + self.A1)
        
        dCl_dalpha = 2 * np.pi
        
        alpha_L0 = alpha - Cl / 2 / np.pi
        
        Cm_LE = -1 * (Cl / 4 + np.pi / 4 * (self.A1 - self.A2))
        
        Cm_c_4 = np.pi / 4 * (self.A2 - self.A1)
        
        return Gamma, Cl, dCl_dalpha, alpha_L0, Cm_LE, Cm_c_4
        

class NACAExternaTAT(ThinAirfoil):
    """
    Loads an airfoil coordinate file and optionally interpolates it to increase/redistribute the number of panels.

    Args:
        sense: "ClockWise" or "CounterClockWise" choose one of those depending on the point order in the file
        path: file path
        interp: grade of the B-spline (0 for no interpolation)
        new_num_p: number of new points for the interpolated curve
        distr: use a non linear point distribution (only cosine-like distribution is implemented at the moment)
    """

    def __init__(self, sense: str, path: str, interp: int = 0, new_num_p=None, distr: int = 0):
        # Load external coordinate file
        data = np.loadtxt(path, dtype=np.float64)

        x = data[:, 0]
        z = data[:, 1]

        # ensure the coordinates are ordered Clockwise.
        if sense == "CounterClockWise":
            x = x[::-1]
            z = z[::-1]
        elif sense == "ClockWise":
            pass
        else:
            raise (ValueError(
                "You must specify whether the set of points is ClockWise or CounterClockWise"))

        # optional geometry refinement using B-Splines
        if interp != 0:
            # group the points in a list of shape (2,N)
            points = [x, z]

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
            x, z = spl(t_new)

        
        #####################################################################
        # Maximum thickness and camber value and locations + Camber line plot
        #####################################################################

        # may not be the faster or precise implementation but it seems indeed straightforward to me

        # find the leading edge index (minimum X coordinate (simplified, to be rigourus we should calculate it as the
        # max distance from the leading edge, but if the points cloud is standard structured it should be fine))
        le_idx = np.argmin(x)

        # slice the arrays based on the Clockwise traversal
        # note: they both share the leading edge point
        # starting from TE (index 0) to LE (le_idx) -> lower surface
        pressure_side = (x[:le_idx + 1], z[:le_idx + 1])

        # from LE (le_idx) back to TE (end of array) -> upper surface
        suction_side = (x[le_idx:], z[le_idx:])

        xs_arr = pressure_side[0]
        zs_arr = pressure_side[1]

        x_t_max = 0
        z_pr_t_max = 0
        z_su_t_max = 0
        t_max = 0
        x_camber_max = 0
        camber_max = 0
        camber_line = []
        for idx, pr_x in enumerate(xs_arr):
            pr_z = zs_arr[idx]

            su_idx = np.abs(suction_side[0] - pr_x).argmin()

            su_x = suction_side[0][su_idx]
            su_z = suction_side[1][su_idx]

            # print(pr_x, pr_y)
            # print(su_x, su_y)

            t = su_z - pr_z

            if t_max < t:
                t_max = t
                x_t_max = pr_x
                y_pr_t_max = pr_z
                y_su_t_max = su_z

            c = 0.5 * (su_z + pr_z)
            camber_line.append(c)
            if camber_max < c:
                camber_max = c
                x_camber_max = pr_x
        


        self.x = x
        self.z = z
        self.t_max = t_max
        self.x_t_max = x_t_max
        self.camber_max = camber_max
        self.x_camber_max = x_camber_max
        super().__init__(xs_arr, np.array(camber_line))
        
        
if __name__ == "__main__":
    #! Note: new_num_p is roughly two times the actual number of points used for calculation due to interpolation 
    #!       algorithm implemented
    NACA = NACAExternaTAT("CounterClockWise", "NACA Airfoils/2026_sup.dat", 
                          interp=3, new_num_p = 1000, distr=0)
    
    plt.gca().axis("equal")
    plt.plot(NACA.x, NACA.z)
    plt.xlabel(r"$\frac{x}{c}\;\mathrm{[-]}$")
    plt.ylabel(r"$\frac{y}{c}\;\mathrm{[-]}$")
    plt.plot(NACA.x_camber, NACA.z_camber, linewidth=0.5, label="camber line")
    plt.legend(fontsize=6)
    if showplots: plt.show()
    
    Gamma, Cl0, dCl_dalpha, alphaL0, Cm_LE, Cm_c_4 = NACA.solve(1, 0 * np.pi / 180)

    print(f"""{"="*80}
{"Airfoil characteristics | Classical thin airfoil theory":^80}
{"="*80}
|{"Geometric properties":^78}|
{"|t_max = ":<20}{NACA.t_max:>18.4f} | {"x_t_max = ":<20}{NACA.x_t_max:>18.4f}|
{"|camber_max = ":<20}{NACA.camber_max:>18.4f} | {"x_camber_max = ":<20}{NACA.x_camber_max:>18.4f}|
|{" ":^78}|
|{"Aerodynamic properties":^78}|
{"|Cl0 = ":<20}{Cl0:>18.4f} | {"dCl_dalpha = ":<20}{dCl_dalpha:>18.3f}|
{"|alpha_L0 = ":<20}{alphaL0 * 180 / np.pi:>14.2f} deg | {" ":<20}{" ":>18}|
{"|Cm_c_4 = ":<20}{Cm_c_4:>18.3e} | {"dCm_c_4_dalpha = ":<20}{0:>18}|
{"|x_ac = ":<20}{0.25:>18.4f} | {" ":<20}{" ":>18}|
{"="*80}""")
