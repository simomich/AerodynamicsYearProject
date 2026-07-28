"""
author:                  Simone Micelli
email:        simonemicelli47@gmail.com
date:                        28/10/2025
description:
    International Standard Atmosphere (ISA) Calculator.
    This module provides a computational model of the Earth's atmosphere based on the
    International Standard Atmosphere (ISA). It calculates standard atmospheric properties
    including temperature, pressure, density, speed of sound, and dynamic viscosity.

    Physical Models & Assumptions:
    - Air modeled as a perfect gas and dry;
    - The gas is in adiabatic conditions;
    - The atmosphere is at rest;
    - Year-round, mid-latitude conditions;

    - Sea level standard conditions: T = 288.15 K, P = 101325 Pa.
    - Constant gravity acceleration: g = 9.80665 m/s^2.
    - Troposphere lapse rate: -6.5 K/km (up to 11,000 m).
    - Stratosphere I: Isothermal at Tropopause temperature (up to 20,000 m).
    - Dynamic Viscosity: Evaluated using Sutherland's law (s = 110.4 K).

    Current limitations:
    - Valid for altitudes from sea level (0 m) up to the end of Stratosphere I (20.000 m).
    - Dynamic viscosity is calculated using Sutherland's law (reliable up to 90 km of altitude).

Copyright (CC) 2025 Simone Micelli
"""

import numpy as np

########################################
# Sea level parameters
########################################
# temperature [K]
T_SL = 288.15
# pressure [Pa]
P_SL = 101325.
# density [kg/m^3]
RHO_SL = 1.225
# speed of sound [m/s]
A_SL = 340.0

########################################
# Physics parameters
########################################
# gravity acceleration (assumed constant with altitude) [m/s^2]
G = 9.80665
# air gas constant [J/kg/K]
R = 287.05287
# adiabatic dilatation coefficient [-]
GAMMA = 1.4

########################################
## Atmospheric limits
########################################
# tropopause height [m]
H_TP = 11000.
# lapse rate [K/m]
dTdh_TS = -6.5 / 1000.
# tropopause temperature [K]
T_TP = T_SL + dTdh_TS * H_TP
# temperature ratio between tropopause & sea level
T_ratio = T_TP / T_SL
# usefull exponent for pressure and density calculations
expP = -G / R / dTdh_TS
# pressure at tropopause [Pa]
press_TP = P_SL * T_ratio ** expP
# density at tropopause [kg/m^3]
rho_TP = RHO_SL * T_ratio ** (expP - 1.)
# end of Stratosphere I [m]
H_SSI = 20000.

########################################
## Class definition
########################################

class ISACondition:
    """
    Calculates atmospheric properties at a specific altitude using the ISA model.

    The properties are evaluated upon instantiation and stored as instance attributes.

    Args:
        h (float): The geometric altitude above sea level in meters.
        t (float): Static temperature [K].
        press (float): Static pressure [Pa].
        rho (float): Air density [kg/m^3].
        a (float): Speed of sound [m/s].
        mu (float): Dynamic viscosity [N*s/m^2], calculated via Sutherland's law.

    Raises:
        ValueError: If the provided altitude `h` is less than 0 or greater than
            the Stratosphere I limit (20,000 m).
    """

    def __init__(self, h: float):
        self.h = h

        # sea level and constants values
        self.T_SL = T_SL
        self.P_SL = P_SL
        self.RHO_SL = RHO_SL
        self.A_SL = A_SL
        self.G = G
        self.R = R
        self.GAMMA = GAMMA

        # h below tropopause
        if 0 <= self.h <= H_TP:
            self.t = T_SL + dTdh_TS * self.h
            t_ratio = self.t / T_SL
            self.press = P_SL * t_ratio ** expP
            self.rho = RHO_SL * t_ratio ** (expP - 1.)

        # h between tropopause and stratosphere I
        elif self.h <= H_SSI:
            self.t = T_TP
            press_ratio = np.exp(-G * (self.h - H_TP) / R / T_TP)
            self.press = press_TP * press_ratio
            self.rho = rho_TP * press_ratio

        # undealed h value
        else:
            raise(ValueError("The given height value isn't handled by the ISA model below the Stratosphere I"))

        # speed of sound at the given height [m/s]
        self.a = A_SL * np.sqrt(self.t / T_SL)

        ########################################
        ## dynamic viscosity by means of Sutherland’s law
        ## reliable up to 90 km of altitude
        ########################################
        # beta coefficient [kg/(s m K^0.5)]
        beta = 1.458e-6
        # Sutherland's constant [K]
        s = 110.4
        # dynamic viscosity [N*s/m^2]
        self.mu = (beta * self.t ** 1.5) / (self.t + s)

        # kinematic viscosity [m^2/s]
        self.nu = self.mu / self.rho

    def __str__(self):
        out = f"""\
Altitude:             h = {self.h:>12.4f}       m
Temperature:          T = {self.t:>12.4f}       K
Pressure:             p = {self.press:>12.4f}      Pa
Density:            rho = {self.rho:>12.4f}  kg/m^3
Speed of sound:       a = {self.a:>12.4f}     m/s
Dynamic viscosity:   mu = {self.mu:>12.4e} N*s/m^2
Kinematic viscosity: nu = {self.nu:>12.4e}   m^2/s"""

        return out

if __name__ == "__main__":
    for i in np.arange(0, 5001, 1000, dtype = float):
        print(ISACondition(i), "-"*50, sep = "\n")

    print(ISACondition(18000), "-"*50, sep = "\n")
