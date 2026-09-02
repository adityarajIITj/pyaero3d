"""
PyAero3D - US Standard Atmosphere 1976 Model (0 to 86km).
Computes pressure, temperature, air density, speed of sound, and dynamic viscosity.
"""

from typing import Tuple, Union
import numpy as np

from pyaero3d.core.types import (
    STANDARD_GRAVITY,
    SEA_LEVEL_PRESSURE,
    SEA_LEVEL_DENSITY,
    SEA_LEVEL_TEMPERATURE,
    AIR_GAS_CONSTANT,
    AIR_HEAT_CAPACITY_RATIO,
    SUTHERLAND_S,
    SUTHERLAND_BETA,
)

# Standard Atmospheric Layer Boundaries (Geopotential Altitude in meters)
LAYER_ALTITUDES = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 86000.0])
# Temperature Lapse Rates L_b in K/m
LAYER_LAPSE_RATES = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])


class StandardAtmosphere:
    """
    Continuous Multi-Layer Atmosphere Model (ICAO / US Standard 1976).
    """

    @staticmethod
    def get_properties(altitude_m: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], ...]:
        """
        Calculates atmospheric state at altitude.
        Returns: (temperature_K, pressure_Pa, density_kg_m3, speed_of_sound_m_s, dynamic_viscosity_Pa_s)
        """
        alt = np.asarray(altitude_m, dtype=np.float64)
        is_scalar = (alt.ndim == 0)

        # Fast analytical branch for standard flight altitudes (0 to 45km)
        alt_clamped = np.clip(alt, 0.0, 86000.0)

        # Troposphere: alt <= 11000m
        T_tropo = SEA_LEVEL_TEMPERATURE - 0.0065 * alt_clamped
        P_tropo = SEA_LEVEL_PRESSURE * (np.maximum(T_tropo / SEA_LEVEL_TEMPERATURE, 1e-4) ** 5.2561)

        # Stratosphere: alt > 11000m
        T_isotherm = 216.65
        P_isotherm = 22632.06 * np.exp(-STANDARD_GRAVITY * (alt_clamped - 11000.0) / (AIR_GAS_CONSTANT * T_isotherm))

        T = np.where(alt_clamped <= 11000.0, T_tropo, T_isotherm)
        P = np.where(alt_clamped <= 11000.0, P_tropo, P_isotherm)

        rho = P / (AIR_GAS_CONSTANT * T)
        a = np.sqrt(AIR_HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT * T)
        mu = (SUTHERLAND_BETA * (T ** 1.5)) / (T + SUTHERLAND_S)

        if is_scalar:
            return float(T), float(P), float(rho), float(a), float(mu)
        return T, P, rho, a, mu
