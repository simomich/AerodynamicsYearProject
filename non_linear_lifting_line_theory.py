"""
author:                  Simone Micelli
email:        simonemicelli47@gmail.com
date:                        27/06/2026
description:
    Educational implementation of the Non-Linear Lifting Line Theory 
    applied to finite wing design and performance evaluation.

Copyright (CC) 2026 Simone Micelli
"""

import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from international_standard_atmosphere import ISACondition


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


########################################
# Airfoil models
########################################

class LinearAirfoil:
    def Cl(self, alpha):
        alpha_l0 = -3*np.pi/180
        return 2*np.pi*(alpha-alpha_l0)


class AirfoilDat:
    def __init__(self, alphas: np.ndarray, cl: np.ndarray, cd: np.ndarray, cdp: np.ndarray, cm: np.ndarray):
        self.alpha_data = alphas
        self.cl_data    = cl
        self.cd_data    = cd
        self.cdp_data   = cdp
        self.cm_data    = cm
        
        self.interpCl    = interp1d(self.alpha_data, self.cl_data)
        self.interpCd    = interp1d(self.alpha_data, self.cd_data)
        self.interpCdp   = interp1d(self.alpha_data, self.cdp_data)
        self.interpCm_c_4 = interp1d(self.alpha_data, self.cm_data)

    @classmethod
    def from_file_dat(cls, filename: str):
        # 1 col angolo incidenza alpha
        # 2 col Cl
        # 3 col Cd
        # 4 col Cdp pressure drag
        # 5 col Cm(c/4)
        data = np.loadtxt(filename, dtype=np.float64)

        return cls(data[:, 0] * np.pi/180,
                   data[:, 1],
                   data[:, 2],
                   data[:, 3],
                   data[:, 4])

    def Cl(self, alpha):
        """
        Returns interpolated Cl for a given alpha [rad].
        """
        return self.interpCl(alpha)

    def Cd(self, alpha):
        """
        Returns interpolated Cd for a given alpha [rad].
        """
        return self.interpCd(alpha)
    
    def Cdp(self, alpha):
        """
        Returns interpolated Cd for a given alpha [rad].
        """
        return self.interpCdp(alpha)

    def Cm_c_4(self, alpha):
        """
        Returns interpolated Cm_c_4 for a given alpha [rad].
        """
        return self.interpCm_c_4(alpha)


class NN_LLT_FiniteWing():
    def __init__(self, airfoil_path, Ns, S, AR, alpha_r, alpha_twist, delta):
        """Creates a NN_LLT_FiniteWing object that represents a finite wing (without dihedral and swept angle)

        Parameters
        ----------
        airfoil_path: str
            file .dat path
        Ns : int
            Number of control points for the numerical method
        S : float
            Projected surface area [m] to the xy plane
        AR : float
            Aspect ratio of the wing
        alpha_r : float
            root geometric angle [rad]
        alpha_twist : float
            twist geometric angle [rad] to calculate c(y)
        delta : float
            taper ratio

        Returns
        -------
            NN_LLT_FiniteWing instance
        """        
        
        # load constant profile (constant)
        #! variable profile not implemented
        self.profile = AirfoilDat.from_file_dat(airfoil_path)
        
        # number of control points to use
        self.Ns = Ns
        # wing area [m^2]
        self.S = S
        # aspect ratio [-]
        self.AR = AR
        # setting angle [rad]
        self.alpha_r = alpha_r
        # geometric twist [rad]
        self.alpha_twist = alpha_twist
        # taper ratio c_t / c_r
        self.delta = delta
        # wing span [m]
        self.b = np.sqrt(AR * S)
        # root chord [m]
        self.c_r = 0.0
        if delta >= 0:
            self.c_r = S * 2 / (self.b * (1 + self.delta))
        else:
            self.c_r = 4 * S / (self.b * np.pi)
        self.c_t = self.c_r * self.delta
        
        
        # il subscript _c considera le grandezze corrispondenti ai centroidi
        self.theta = np.linspace(0, np.pi, Ns + 1)
        self.theta_c = self.theta[:-1] + np.diff(self.theta) * 0.5

        self.y = 0.5 * self.b * np.cos(self.theta)
        self.dy = np.diff(self.y)
        self.y_c = 0.5 * self.b * np.cos(self.theta_c)

        if self.delta >= 0:
            self.c = self.c_r * (1 - (1 - self.delta) * np.fabs(np.cos(self.theta)))
            self.c_c = self.c_r * (1 - (1 - self.delta) * np.fabs(np.cos(self.theta_c)))
        else:
            self.c = self.c_r * np.sqrt(1 - np.cos(self.theta)**2)
            self.c_c = self.c_r * np.sqrt(1 - np.cos(self.theta_c)**2)

        self.alpha_c = self.alpha_r + self.alpha_twist * np.fabs(np.cos(self.theta_c))
        
        alpha_r_deg = self.alpha_r*180/np.pi
        alpha_twist_deg = self.alpha_twist*180/np.pi
        print(f"""{"="*80}\n{"NL-LLT":^80}\n{"="*80}
|{"Geometric properties":^78}|
{"|S = ":<20}{self.S:>14.2f} m^2 | {"AR = ":<20}{self.AR:>18.2f}|
{"|b = ":<20}{self.b:>16.2f} m | {"Taper ratio = ":<20}{self.delta:>18.2f}|
{"|c_r = ":<20}{self.c_r:>16.3f} m | {"c_t = ":<20}{self.c_t:>16.3f} m|
{"|alpha_r (guess) = ":<24}{alpha_r_deg:>10.2f} deg | {"alpha_twist = ":<20}{alpha_twist_deg:>14.2f} deg|
|{" ":^78}|""")
    
    
    def getAlpha_i(self, An_f, theta_f):
        return np.array([np.sum([(n+1) * a * np.sin((n+1) * t)
                                for n, a in enumerate(An_f)]) / np.sin(t) for t in theta_f])

    def getGamma(self, An_f, theta_f):
        return 2 * self.b * Vinf * \
                np.array([np.sum([a * np.sin((n+1) * t) for n, a in enumerate(An_f)]) for t in theta_f])
    
    def res(self, x):
        alpha_i = self.getAlpha_i(x, self.theta_c)
        alpha_e = self.alpha_c - alpha_i

        cl = self.profile.Cl(alpha_e)
        Gamma1 = cl * Vinf * self.c_c/2
        Gamma2 = self.getGamma(x, self.theta_c)

        return Gamma1 - Gamma2
        
    def solve(self, Vinf, RHOinf):
        """Solves the non linear system for the Fourier series coefficients of Gamma

        Parameters
        ----------
        Vinf : float
            Freestream velocity
        RHOinf : float
            Density of the fluid
            
        Returns
        -------
        np.array
            Array filled with the Fourier series coefficients of Gamma
        """
        An = np.zeros(self.Ns)
        An = fsolve(self.res, An)
        return An
    
    def solve_isa(self, Vinf, altitude):
        """Solve the flow with isa condition

        Parameters
        ----------
        Vinf : float
            Freestream velocity
        altitude : float
            altitude for isa calculations
           
        Returns
        -------
            The solution but with Vinf + Altitude approach
        """
        isa = ISACondition(altitude)
        RHOinf = isa.rho
        return self.solve(Vinf, RHOinf)
    
    def target_lift_calculation(self, L_target, Vinf, altitude):
        """Finds a certain CL where the only free variable is alpha_r

        Parameters
        ----------
        L_target : float
            The targeted Lift value in [N]
        Vinf : float
            Freestream velocity in [m/s]
        altitude : float
            altitude in [m] for estimating air condition using ISA

        Returns
        -------
        np.array
            Array filled with the Fourier series coefficients of Gamma based on the optimized version of self.alpha_r
        """
        isa = ISACondition(altitude)
        
        def lift_residual(alpha_guess):
            self.alpha_r = alpha_guess[0]
            self.alpha_c = self.alpha_r + self.alpha_twist * np.fabs(np.cos(self.theta_c))
            
            An = self.solve(Vinf, isa.rho)
            Gamma = self.getGamma(An, self.theta_c)
            L = np.sum(isa.rho * Vinf * Gamma * np.abs(self.dy))
            
            return L - L_target
        
        print(f"""|{"alpha_r calculation":^78}|""")
        
        initial_guess = self.alpha_r
        alpha_opt = fsolve(lift_residual, [initial_guess])[0]
        
        self.alpha_r = alpha_opt
        self.alpha_c = self.alpha_r + self.alpha_twist * np.fabs(np.cos(self.theta_c))
        An_opt = self.solve(Vinf, isa.rho)
        
        print(f"""{"|Required alpha_r = ":<20}{self.alpha_r*180/np.pi:>14.2f} deg | {" ":<20}{" ":>18}|
|{" ":^78}|""")
        
        return An_opt


def print_isa_conditions(altitude):
    """Print the International Standard Atmosphere's condition in the usual table format

    Parameters
    ----------
    altitude : float
        Altitude at which we estimate air condition by means of ISA
    """
    isa = ISACondition(altitude)
    print(f"""\n{'='*80}\n{"Flight conditions":^80}\n{'='*80}
{"|Altitude = ":<20}{altitude:>16.0f} m | {"T = ":<20}{isa.t:>16.2f} K|
{"|p = ":<20}{isa.press:>15.0f} Pa | {"rho = ":<20}{isa.rho:>11.4f} kg/m^3|
{"|a = ":<20}{isa.a:>14.1f} m/s | {"mu = ":<19}{isa.mu:>9.4e} kg/(m*s)|
{"|nu = ":<20}{isa.ni:>12.4e} m^2/s | {" ":<20}{" ":>18}|
{'='*80}""")


def evaluate_wing_performance(wing: NN_LLT_FiniteWing, An, Vinf, altitude, verbose=True):
    """Computes the aerodynamic performance and structural loads of the wing
    using Simpson's numerical integration method.

    Parameters
    ----------
    wing : NN_LLT_FiniteWing
        Instance of the class NN_LLT_FiniteWing
    An : np.nparray
        Array containing the Fourier series coefficients for Gamma
    Vinf : float
        Freestream velocity [m/s]
    altitude : float
        Altitude value [m]
    verbose : bool
        print the results in console?

    Returns
    -------
    dict
        Results:
        - Total Lift
        - Total induced Drag
        - Lift coefficient
        - Induced drag coefficient
        - Oswald's factor
        - Total pitching moment value
        - Maximum bending moment (root)
        - Maximum torque (torsional, root)
    """
    isa = ISACondition(altitude)
    RHOinf = isa.rho
    q_inf = 0.5 * RHOinf * Vinf**2
    
    # prepare ascending arrays (from -b/2 to b/2) for Simpson's rule integration
    y_c_asc = wing.y_c[::-1]
    c_c_asc = wing.c_c[::-1]
    
    ma_chord = simpson(c_c_asc**2, y_c_asc) / wing.S
    if verbose: print(f"""{"|m_a_chord = ":<20}{ma_chord:>16.3f} m | {" ":<20}{" ":>18}|
{"|Re = ":<20}{Vinf * ma_chord / isa.ni:>18.3e} | {" ":<20}{" ":>18}|
|{" ":^78}|""")
    
    alpha_i = wing.getAlpha_i(An, wing.theta_c)
    alpha_e = wing.alpha_c - alpha_i
    alpha_e_asc = alpha_e[::-1]
    
    Gamma = wing.getGamma(An, wing.theta_c)
    Gamma_asc = Gamma[::-1]

    # Total Lift (L) calculation using Simpson's rule
    integrand_L = RHOinf * Vinf * Gamma_asc
    L = simpson(y=integrand_L, x=y_c_asc)

    CL = np.pi * wing.AR * An[0]
    CDi = np.pi * wing.AR * sum([(n+1) * a**2 for n, a in enumerate(An)])
    e = 1.0 / (CDi * np.pi * wing.AR / CL**2)

    # Total Aerodynamic Pitching Moment
    Cm_local_asc = wing.profile.Cm_c_4(alpha_e_asc)
    integrand_pitch = q_inf * (c_c_asc**2) * Cm_local_asc
    Total_Pitching_Moment = simpson(y=integrand_pitch, x=y_c_asc)

    # Root Loads
    mask_half = y_c_asc > 0
    y_half = y_c_asc[mask_half]
    Gamma_half = Gamma_asc[mask_half]
    integrand_pitch_half = integrand_pitch[mask_half]

    # Maximum Root Bending Moment
    integrand_bend = RHOinf * Vinf * Gamma_half * y_half
    Root_Bending_Moment = simpson(y=integrand_bend, x=y_half)

    # Maximum Root Torsional Moment
    Root_Torsional_Moment = simpson(y=integrand_pitch_half, x=y_half)

    # Induced Drag
    #TODO: Implement a boundary layer solution on the airfoil in order to estimate the skin friction drag
    #TODO: Or another possibility is the laminar wing distribution of tau_wall but it may consider the turbulent
    #TODO: boundary layer as it will develop more than just laminar
    Total_Induced_Drag = q_inf * wing.S * CDi

    # print results only if verbose is True
    if verbose:
        print(f"""|{"Aerodynamic performance":^78}|
{"|L = ":<20}{L:>16.1f} N | {"CL = ":<20}{CL:>18.4f}|
{"|CDi = ":<20}{CDi:>18.4e} | {"e = ":<20}{e:>18.4f}|
{"="*80}""")
        
        print(f"""|{"Structural Loads & Drag":^78}|
{"|Total Pitching Moment = ":<25}{Total_Pitching_Moment:>10.1f} Nm | {" ":<20}{" ":>18}|
{"|Root Bending Moment = ":<25}{Root_Bending_Moment:>10.1f} Nm | {" ":<20}{" ":>18}|
{"|Root Torsion Moment = ":<25}{Root_Torsional_Moment:>10.1f} Nm | {" ":<20}{" ":>18}|
{"|Total Drag (Induced) = ":<25}{Total_Induced_Drag:>10.1f} N  | {" ":<20}{" ":>18}|
{"="*80}""")

    return {
        "L": L, "CL": CL, "CDi": CDi, "e": e,
        "Total_Pitching_Moment": Total_Pitching_Moment,
        "Root_Bending_Moment": Root_Bending_Moment,
        "Root_Torsional_Moment": Root_Torsional_Moment,
        "Total_Induced_Drag": Total_Induced_Drag,
    }


if __name__ == "__main__":
    ########################################
    # Parameters
    ########################################
    Vinf = 40.0               # [m/s]
    ALTITUDE = 5000           # [m]
    TARGET_LIFT = 300.0       # [kg]
    
    # conversion
    TARGET_LIFT *= 9.80665    # [N]
    
    isa = ISACondition(ALTITUDE)
    RHOinf = isa.rho
    
    # Wing shared geometric properties
    b              = 13          # [m]
    AR             = 11          # [-]
    S              = b**2 / AR   # [m^2]
    
    print_isa_conditions(ALTITUDE)
    
    airfoil_file = "airfoilsdat/2026_performance.dat"

    ########################################
    # Wing 1 (Baseline Tapered)
    ########################################
    print(f"\n{'='*80}\n{'WING 1: Baseline Tapered (delta=0.5, twist=-1.0 deg)':^80}\n{'='*80}")
    g_wing1 = {
        "Ns": 50, "S": S, "AR": AR,
        "alpha_r": 2.0 * np.pi / 180, 
        "alpha_twist": -1.0 * np.pi / 180, 
        "delta": 0.5
    }
    wing1 = NN_LLT_FiniteWing(airfoil_file, **g_wing1)
    An1 = wing1.target_lift_calculation(TARGET_LIFT, Vinf, ALTITUDE)
    res1 = evaluate_wing_performance(wing1, An1, Vinf, ALTITUDE)
    wing1_design_alpha_r = wing1.alpha_r
    wing1_design_L = res1["L"]
    wing1_design_Di = res1["Total_Induced_Drag"]
    wing1_design_M = res1["Total_Pitching_Moment"]

    ########################################
    # Wing 2 (Rectangular)
    ########################################
    print(f"\n{'='*80}\n{'WING 2: Rectangular (delta=1.0, twist=0.0 deg)':^80}\n{'='*80}")
    g_wing2 = {
        "Ns": 50, "S": S, "AR": AR,
        "alpha_r": 2.0 * np.pi / 180, 
        "alpha_twist": 0.0 * np.pi / 180, 
        "delta": 1.0
    }
    wing2 = NN_LLT_FiniteWing(airfoil_file, **g_wing2)
    An2 = wing2.target_lift_calculation(TARGET_LIFT, Vinf, ALTITUDE)
    res2 = evaluate_wing_performance(wing2, An2, Vinf, ALTITUDE)

    ########################################
    # Wing 3 (More Tapered)
    ########################################
    print(f"\n{'='*80}\n{'WING 3: More Tapered (delta=0.4, twist=0.0 deg)':^80}\n{'='*80}")
    g_wing3 = {
        "Ns": 50, "S": S, "AR": AR,
        "alpha_r": 2.0 * np.pi / 180, 
        "alpha_twist": 0.0 * np.pi / 180, 
        "delta": 0.4
    }
    wing3 = NN_LLT_FiniteWing(airfoil_file, **g_wing3)
    An3 = wing3.target_lift_calculation(TARGET_LIFT, Vinf, ALTITUDE)
    res3 = evaluate_wing_performance(wing3, An3, Vinf, ALTITUDE)

    ########################################
    # Wing 4 (Elliptical)
    ########################################
    print(f"\n{'='*80}\n{'WING 4: Elliptical (twist=0.0 deg)':^80}\n{'='*80}")
    g_wing4 = {
        "Ns": 50, "S": S, "AR": AR,
        "alpha_r": 2.0 * np.pi / 180, 
        "alpha_twist": 0.0 * np.pi / 180, 
        "delta": -1.0 # Triggers elliptical profile
    }
    wing4 = NN_LLT_FiniteWing(airfoil_file, **g_wing4)
    An4 = wing4.target_lift_calculation(TARGET_LIFT, Vinf, ALTITUDE)
    res4 = evaluate_wing_performance(wing4, An4, Vinf, ALTITUDE)


    ########################################
    # Sweep of alpha_r for Wing 1
    ########################################
    print(f"\n{'='*80}\n{'WING 1: Characteristic Curves (Sweep alpha_r from 0 to 6 deg)':^80}\n{'='*80}")
    alphas_sweep_deg = np.linspace(-6, 6, 50)
    L_sweep = []
    Di_sweep = []
    M_sweep = []

    for a_deg in alphas_sweep_deg:
        # for each alphas in the array we solve the wing
        wing1.alpha_r = a_deg * np.pi / 180
        wing1.alpha_c = wing1.alpha_r + wing1.alpha_twist * np.fabs(np.cos(wing1.theta_c))
        An_sweep = wing1.solve_isa(Vinf, ALTITUDE)
        res_sweep = evaluate_wing_performance(wing1, An_sweep, Vinf, ALTITUDE, verbose=False)
        
        L_sweep.append(res_sweep["L"])
        Di_sweep.append(res_sweep["Total_Induced_Drag"])
        M_sweep.append(res_sweep["Total_Pitching_Moment"])

    ########################################
    # Comparative Plots & Sweep Plots
    ########################################
    if showplots:
        # Plot 1: Planform Comparison
        plt.plot(wing1.y, wing1.c, label=r"Wing 1 (Tapered $\lambda=0.5$)", color="blue")
        plt.plot(wing2.y, wing2.c, label=r"Wing 2 (Rectangular)", color="red", alpha=0.4)
        plt.plot(wing3.y, wing3.c, label=r"Wing 3 (More Tapered $\lambda=0.4$)", color="green", alpha=0.4)
        plt.plot(wing4.y, wing4.c, label=r"Wing 4 (Elliptical)", color="orange", alpha=0.4)
        
        plt.axis("equal") 
        plt.xlabel("$y$ [m]")
        plt.ylabel("$c(y)$ [m]")
        plt.legend(loc="lower right", fontsize=6)
        if not savefigures: plt.title("Wing planforms comparison")
        else: plt.savefig("saved_plots/wing_planforms_comparison.pdf", format="pdf")
        plt.show()

        # Plot 2: Local Cl Comparison
        Clplot1 = 2 * wing1.getGamma(An1, wing1.theta_c) / (Vinf * wing1.c_c)
        Clplot2 = 2 * wing2.getGamma(An2, wing2.theta_c) / (Vinf * wing2.c_c)
        Clplot3 = 2 * wing3.getGamma(An3, wing3.theta_c) / (Vinf * wing3.c_c)
        Clplot4 = 2 * wing4.getGamma(An4, wing4.theta_c) / (Vinf * wing4.c_c)
        
        plt.plot(wing1.y_c, Clplot1, label=r"Wing 1 (Tapered $\lambda=0.5$)", color="blue")
        plt.plot(wing2.y_c, Clplot2, label=r"Wing 2 (Rectangular)", color="red", alpha=0.4)
        plt.plot(wing3.y_c, Clplot3, label=r"Wing 3 (More Tapered $\lambda=0.4$)", color="green", alpha=0.4)
        plt.plot(wing4.y_c, Clplot4, label=r"Wing 4 (Elliptical)", color="orange", alpha=0.4)
        
        plt.xlabel("$y$ [m]")
        plt.ylabel("$C_L$ [-]")
        plt.legend(fontsize=6)
        if not savefigures: plt.title("Spanwise $C_L$ Comparison")
        else: plt.savefig("saved_plots/spanwise_CL_comparison.pdf", format="pdf")
        plt.show()
        
        # Plot 3: Induced Angle of Attack Comparison
        alpha_i1_deg = wing1.getAlpha_i(An1, wing1.theta_c) * 180/np.pi
        alpha_i2_deg = wing2.getAlpha_i(An2, wing2.theta_c) * 180/np.pi
        alpha_i3_deg = wing3.getAlpha_i(An3, wing3.theta_c) * 180/np.pi
        alpha_i4_deg = wing4.getAlpha_i(An4, wing4.theta_c) * 180/np.pi
        
        plt.plot(wing1.y_c, alpha_i1_deg, label=r"Wing 1 (Tapered $\lambda=0.5$)", color="blue")
        plt.plot(wing2.y_c, alpha_i2_deg, label=r"Wing 2 (Rectangular)", color="red", alpha=0.4)
        plt.plot(wing3.y_c, alpha_i3_deg, label=r"Wing 3 (More Tapered $\lambda=0.4$)", color="green", alpha=0.4)
        plt.plot(wing4.y_c, alpha_i4_deg, label=r"Wing 4 (Elliptical)", color="orange", alpha=0.4)
        
        plt.xlabel("$y$ [m]")
        plt.ylabel(r"$\alpha_i$ [deg]")
        plt.legend(fontsize=6)
        if not savefigures: plt.title(r"Spanwise ($\alpha_i$) Comparison")
        else: plt.savefig("saved_plots/spanwise_alpha_i_comparison.pdf", format="pdf")
        plt.show()

        # Plot 4: Wing 1 Global Lift vs alpha_r
        plt.plot(alphas_sweep_deg, L_sweep, color="blue")
        plt.scatter(wing1_design_alpha_r * 180 / np.pi, wing1_design_L, 
                    color="#8900F1", s=5, zorder=50, label="Design Point $L$ = 300 [kg]")
        plt.legend(fontsize=6)
        plt.xlabel(r"$\alpha_r$ [deg]")
        plt.ylabel("L [N]")
        if not savefigures: plt.title(r"Wing 1: L vs $\alpha_r$")
        else: plt.savefig("saved_plots/wing1_L_vs_alpha_r.pdf", format="pdf")
        plt.show()

        # Plot 5: Wing 1 Induced Drag vs alpha_r
        plt.plot(alphas_sweep_deg, Di_sweep, color="red")
        plt.scatter(wing1_design_alpha_r * 180 / np.pi, wing1_design_Di, 
                    color="#8900F1", s=5, zorder=50, label="Design Point")
        plt.legend(fontsize=6)
        plt.xlabel(r"$\alpha_r$ [deg]")
        plt.ylabel(r"$D_i$ [N]")
        if not savefigures: plt.title(r"Wing 1: $D_i$ vs $\alpha_r$")
        else: plt.savefig("saved_plots/wing1_D_i_vs_alpha_r.pdf", format="pdf")
        plt.show()
        
        # Plot 6: Wing 1 Total Pitching Moment vs alpha_r
        plt.plot(alphas_sweep_deg, M_sweep, color="green")
        plt.scatter(wing1_design_alpha_r * 180 / np.pi, wing1_design_M, 
                    color="#8900F1", s=5, zorder=50, label="Design Point")
        plt.legend(fontsize=6)
        plt.xlabel(r"$\alpha_r$ [deg]")
        plt.ylabel(r"$M$ [Nm]")
        if not savefigures: plt.title(r"Wing 1: $M$ vs $\alpha_r$")
        else: plt.savefig("saved_plots/wing1_M_vs_alpha_r.pdf", format="pdf")
        plt.show()
