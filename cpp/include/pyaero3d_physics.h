/**
 * PyAero3D - Native C++ Multi-Body Aerospace & Orbital Physics Core.
 * Header: pyaero3d_physics.h
 *
 * Provides high-frequency 1000Hz-2000Hz vector physics, quaternion kinematics,
 * US 1976 Standard Atmosphere, 6-DOF aerodynamics, and symplectic integration.
 */

#ifndef PYAERO3D_PHYSICS_H
#define PYAERO3D_PHYSICS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define PYAERO3D_API __declspec(dllexport)
#else
#define PYAERO3D_API __attribute__((visibility("default")))
#endif

#define NUM_FIELDS 32
#define G_GRAV 6.67430e-11
#define G_STANDARD 9.80665

// 32-Stride State Buffer Field Offsets
enum StateStride {
    OFFSET_MASS = 0,
    OFFSET_PX = 1, OFFSET_PY = 2, OFFSET_PZ = 3,
    OFFSET_VX = 4, OFFSET_VY = 5, OFFSET_VZ = 6,
    OFFSET_RADIUS = 7,
    OFFSET_TYPE = 8,
    OFFSET_CD = 9,
    OFFSET_AREA = 10,
    OFFSET_QW = 11, OFFSET_QX = 12, OFFSET_QY = 13, OFFSET_QZ = 14,
    OFFSET_WX = 15, OFFSET_WY = 16, OFFSET_WZ = 17,
    OFFSET_FX = 18, OFFSET_FY = 19, OFFSET_FZ = 20,
    OFFSET_TX = 21, OFFSET_TY = 22, OFFSET_TZ = 23,
    OFFSET_ACTIVE = 24,
    OFFSET_THROTTLE = 25,
    OFFSET_CTRL_ELEV = 26,
    OFFSET_CTRL_AIL = 27,
    OFFSET_CTRL_RUD = 28,
    OFFSET_FUEL_MASS = 29,
    OFFSET_SURFACE_FRIC = 30,
    OFFSET_ON_GROUND = 31
};

/**
 * Evaluates US 1976 Standard Atmosphere at geopotential altitude.
 * @param alt_m Altitude in meters above sea level
 * @param out_rho Output air density (kg/m^3)
 * @param out_p Output pressure (Pa)
 * @param out_t Output temperature (K)
 * @param out_a Output speed of sound (m/s)
 */
PYAERO3D_API void aero_atmosphere_evaluate_c(
    double alt_m,
    double* out_rho,
    double* out_p,
    double* out_t,
    double* out_a
);

/**
 * Quaternion Rodrigues rotation: v_out = q * v * q^-1.
 */
PYAERO3D_API void quaternion_rotate_vector_c(
    const double q[4],
    const double v[3],
    double v_out[3]
);

/**
 * Symplectic Leapfrog Integration Step for N entities.
 * @param state_tensor Pointer to flat contiguous double array of shape (N * 32)
 * @param num_entities Total number of entity slots in buffer
 * @param dt Integration time step in seconds (e.g. 0.001 for 1000Hz)
 */
PYAERO3D_API void physics_step_leapfrog_c(
    double* state_tensor,
    int num_entities,
    double dt
);

/**
 * Vectorized Pairwise N-Body Gravitational Forces in C++.
 */
PYAERO3D_API void compute_nbody_gravity_forces_c(
    double* state_tensor,
    int num_entities
);

/**
 * Class 11 Kinetic Breakup Engine: Exact Zero-Residual Linear Momentum Balance.
 */
PYAERO3D_API void fragment_entity_class11_c(
    double* state_tensor,
    int parent_idx,
    int* shard_indices,
    int num_shards,
    double dispersion_energy
);

#ifdef __cplusplus
}
#endif

#endif // PYAERO3D_PHYSICS_H
