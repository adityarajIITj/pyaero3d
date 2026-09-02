"""
PyAero3D - Physical Constants, Entity Enums, and Contiguous 32-Stride Tensor Indices.
"""

from enum import IntEnum
import numpy as np

# Universal Physical Constants (CODATA 2018 / ICAO Standard Atmosphere)
G_GRAVITATIONAL = 6.67430e-11   # Gravitational constant (m^3 / (kg s^2))
EARTH_MASS = 5.97219e24         # Earth Mass (kg)
EARTH_RADIUS = 6371000.0        # Earth Mean Radius (m)
STANDARD_GRAVITY = 9.80665      # Standard Sea-Level Gravity (m/s^2)
SEA_LEVEL_PRESSURE = 101325.0   # Standard Sea-Level Pressure (Pa)
SEA_LEVEL_DENSITY = 1.2250      # Standard Sea-Level Air Density (kg/m^3)
SEA_LEVEL_TEMPERATURE = 288.15  # Standard Sea-Level Temperature (K)
AIR_GAS_CONSTANT = 287.05287    # Specific Gas Constant for Dry Air (J / (kg K))
AIR_HEAT_CAPACITY_RATIO = 1.40  # Ratio of Specific Heats (gamma)
SUTHERLAND_S = 110.4            # Sutherland Constant for Dynamic Viscosity (K)
SUTHERLAND_BETA = 1.458e-6      # Sutherland Beta Constant (kg / (m s K^0.5))


class StateIdx(IntEnum):
    """
    Contiguous 32-Field Layout per Entity in StateBuffer.
    Shape: (N, 32), float64
    """
    MASS = 0            # Total mass in kg (structural + fuel)
    PX = 1              # Position X (m, Global Viewport: X = East/Right)
    PY = 2              # Position Y (m, Global Viewport: Y = Altitude Up)
    PZ = 3              # Position Z (m, Global Viewport: Z = North/Forward)
    VX = 4              # Velocity X (m/s)
    VY = 5              # Velocity Y (m/s)
    VZ = 6              # Velocity Z (m/s)
    RADIUS = 7          # Bounding collision radius (m)
    ENTITY_TYPE = 8     # EntityType enum integer
    CD = 9              # Base drag coefficient
    AREA = 10           # Aerodynamic reference area (m^2)
    QW = 11             # Quaternion scalar w (attitude)
    QX = 12             # Quaternion vector x
    QY = 13             # Quaternion vector y
    QZ = 14             # Quaternion vector z
    WX = 15             # Angular velocity omega_x (rad/s, body pitch/roll/yaw)
    WY = 16             # Angular velocity omega_y (rad/s)
    WZ = 17             # Angular velocity omega_z (rad/s)
    FX = 18             # Accumulated Force X (N)
    FY = 19             # Accumulated Force Y (N)
    FZ = 20             # Accumulated Force Z (N)
    TX = 21             # Accumulated Torque X (N*m)
    TY = 22             # Accumulated Torque Y (N*m)
    TZ = 23             # Accumulated Torque Z (N*m)
    ACTIVE = 24         # 1.0 if allocated, 0.0 if free slot in pool
    THROTTLE = 25       # Throttle command [0.0 to 1.0]
    CTRL_ELEVATOR = 26  # Elevator control surface [-1.0 to +1.0]
    CTRL_AILERON = 27   # Aileron control surface [-1.0 to +1.0]
    CTRL_RUDDER = 28    # Rudder control surface [-1.0 to +1.0]
    FUEL_MASS = 29      # Remaining propellant mass (kg)
    SURFACE_FRICTION = 30 # Ground tire/surface friction coefficient
    ON_GROUND = 31      # 1.0 if in ground contact, 0.0 if airborne


class EntityType(IntEnum):
    """Supported multi-domain aerospace entity types."""
    FREE_PARTICLE = 0
    FIXED_WING_JET = 1
    QUADROTOR_DRONE = 2
    CARGO_PARACHUTE = 3
    MULTI_STAGE_ROCKET = 4
    DEBRIS_FRAGMENT = 5
    TERRAIN_ANCHOR = 6


# Stride length for tensor memory allocation
STRIDE_LEN = 32
