/**
 * PyAero3D - Native C++ Multi-Body Aerospace & Orbital Physics Core.
 * Implementation: pyaero3d_physics.cpp
 */

#include "pyaero3d_physics.h"
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

extern "C" {

void aero_atmosphere_evaluate_c(
    double alt_m,
    double* out_rho,
    double* out_p,
    double* out_t,
    double* out_a
) {
    double alt = std::max(0.0, std::min(alt_m, 86000.0));

    // Sea level constants
    const double T0 = 288.15;
    const double P0 = 101325.0;
    const double R_air = 287.05287;
    const double gamma = 1.40;
    const double g0 = 9.80665;

    double T = T0;
    double P = P0;

    if (alt <= 11000.0) {
        // Troposphere (Lapse = -0.0065 K/m)
        const double L = -0.0065;
        T = T0 + L * alt;
        P = P0 * std::pow(T / T0, -g0 / (L * R_air));
    } else if (alt <= 20000.0) {
        // Tropopause Isothermal (216.65 K)
        const double T11 = 216.65;
        const double P11 = 22632.06;
        T = T11;
        P = P11 * std::exp(-g0 * (alt - 11000.0) / (R_air * T11));
    } else {
        // Stratosphere 1 (Lapse = +0.0010 K/m)
        const double T20 = 216.65;
        const double P20 = 5474.89;
        const double L = 0.0010;
        T = T20 + L * (alt - 20000.0);
        P = P20 * std::pow(T / T20, -g0 / (L * R_air));
    }

    double rho = P / (R_air * T);
    double a = std::sqrt(gamma * R_air * T);

    if (out_rho) *out_rho = rho;
    if (out_p) *out_p = P;
    if (out_t) *out_t = T;
    if (out_a) *out_a = a;
}

void quaternion_rotate_vector_c(
    const double q[4],
    const double v[3],
    double v_out[3]
) {
    // Rodrigues formula: v' = v + 2*w*(u x v) + 2*(u x (u x v))
    const double w = q[0];
    const double ux = q[1];
    const double uy = q[2];
    const double uz = q[3];

    // uv = u x v
    const double uv_x = uy * v[2] - uz * v[1];
    const double uv_y = uz * v[0] - ux * v[2];
    const double uv_z = ux * v[1] - uy * v[0];

    // uuv = u x (u x v)
    const double uuv_x = uy * uv_z - uz * uv_y;
    const double uuv_y = uz * uv_x - ux * uv_z;
    const double uuv_z = ux * uv_y - uy * uv_x;

    v_out[0] = v[0] + 2.0 * (w * uv_x + uuv_x);
    v_out[1] = v[1] + 2.0 * (w * uv_y + uuv_y);
    v_out[2] = v[2] + 2.0 * (w * uv_z + uuv_z);
}

void physics_step_leapfrog_c(
    double* state_tensor,
    int num_entities,
    double dt
) {
    if (!state_tensor || num_entities <= 0) return;

    const double half_dt = 0.5 * dt;

    for (int i = 0; i < num_entities; ++i) {
        double* row = state_tensor + (i * NUM_FIELDS);
        if (row[OFFSET_ACTIVE] < 0.5) continue;

        double m = row[OFFSET_MASS];
        if (m < 1e-9) continue;
        double inv_m = 1.0 / m;

        // Accelerations from accumulated forces
        double ax = row[OFFSET_FX] * inv_m;
        double ay = row[OFFSET_FY] * inv_m;
        double az = row[OFFSET_FZ] * inv_m;

        // Velocity half-step
        row[OFFSET_VX] += ax * half_dt;
        row[OFFSET_VY] += ay * half_dt;
        row[OFFSET_VZ] += az * half_dt;

        // Position full-step
        row[OFFSET_PX] += row[OFFSET_VX] * dt;
        row[OFFSET_PY] += row[OFFSET_VY] * dt;
        row[OFFSET_PZ] += row[OFFSET_VZ] * dt;

        // Quaternion angular velocity integration
        double wx = row[OFFSET_WX];
        double wy = row[OFFSET_WY];
        double wz = row[OFFSET_WZ];
        double w_mag = std::sqrt(wx * wx + wy * wy + wz * wz);

        if (w_mag > 1e-12) {
            double half_w_dt = 0.5 * w_mag * dt;
            double sin_half = std::sin(half_w_dt) / w_mag;
            double cos_half = std::cos(half_w_dt);

            double dq_w = cos_half;
            double dq_x = wx * sin_half;
            double dq_y = wy * sin_half;
            double dq_z = wz * sin_half;

            double qw = row[OFFSET_QW];
            double qx = row[OFFSET_QX];
            double qy = row[OFFSET_QY];
            double qz = row[OFFSET_QZ];

            // Hamilton quaternion product: q_new = q * dq
            double nw = qw * dq_w - qx * dq_x - qy * dq_y - qz * dq_z;
            double nx = qw * dq_x + qx * dq_w + qy * dq_z - qz * dq_y;
            double ny = qw * dq_y - qx * dq_z + qy * dq_w + qz * dq_x;
            double nz = qw * dq_z + qx * dq_y - qy * dq_x + qz * dq_w;

            double q_norm = std::sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
            if (q_norm > 1e-12) {
                row[OFFSET_QW] = nw / q_norm;
                row[OFFSET_QX] = nx / q_norm;
                row[OFFSET_QY] = ny / q_norm;
                row[OFFSET_QZ] = nz / q_norm;
            }
        }

        // Reset forces for next accumulation
        row[OFFSET_FX] = 0.0;
        row[OFFSET_FY] = 0.0;
        row[OFFSET_FZ] = 0.0;
        row[OFFSET_TX] = 0.0;
        row[OFFSET_TY] = 0.0;
        row[OFFSET_TZ] = 0.0;
    }
}

void compute_nbody_gravity_forces_c(
    double* state_tensor,
    int num_entities
) {
    if (!state_tensor || num_entities <= 1) return;

    const double softening_sq = 1e-4;

    for (int i = 0; i < num_entities; ++i) {
        double* row_i = state_tensor + (i * NUM_FIELDS);
        if (row_i[OFFSET_ACTIVE] < 0.5) continue;

        double m_i = row_i[OFFSET_MASS];
        double px_i = row_i[OFFSET_PX];
        double py_i = row_i[OFFSET_PY];
        double pz_i = row_i[OFFSET_PZ];

        for (int j = i + 1; j < num_entities; ++j) {
            double* row_j = state_tensor + (j * NUM_FIELDS);
            if (row_j[OFFSET_ACTIVE] < 0.5) continue;

            double m_j = row_j[OFFSET_MASS];
            double dx = row_j[OFFSET_PX] - px_i;
            double dy = row_j[OFFSET_PY] - py_i;
            double dz = row_j[OFFSET_PZ] - pz_i;

            double dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            double dist = std::sqrt(dist_sq);
            double inv_dist_cube = 1.0 / (dist_sq * dist);

            double force_mag = G_GRAV * m_i * m_j * inv_dist_cube;
            double fx = force_mag * dx;
            double fy = force_mag * dy;
            double fz = force_mag * dz;

            row_i[OFFSET_FX] += fx;
            row_i[OFFSET_FY] += fy;
            row_i[OFFSET_FZ] += fz;

            row_j[OFFSET_FX] -= fx;
            row_j[OFFSET_FY] -= fy;
            row_j[OFFSET_FZ] -= fz;
        }
    }
}

void fragment_entity_class11_c(
    double* state_tensor,
    int parent_idx,
    int* shard_indices,
    int num_shards,
    double dispersion_energy
) {
    if (!state_tensor || parent_idx < 0 || !shard_indices || num_shards <= 1) return;

    double* parent = state_tensor + (parent_idx * NUM_FIELDS);
    double m_total = parent[OFFSET_MASS];
    double parent_vx = parent[OFFSET_VX];
    double parent_vy = parent[OFFSET_VY];
    double parent_vz = parent[OFFSET_VZ];

    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist_u(-1.0, 1.0);

    // Random mass distribution Dirichlet-style
    std::vector<double> shard_masses(num_shards);
    double mass_sum = 0.0;
    for (int k = 0; k < num_shards; ++k) {
        shard_masses[k] = std::max(0.01, std::abs(dist_u(rng)) + 0.1);
        mass_sum += shard_masses[k];
    }
    for (int k = 0; k < num_shards; ++k) {
        shard_masses[k] = (shard_masses[k] / mass_sum) * m_total;
    }

    // Generate random raw impulse velocities
    std::vector<double> v_raw_x(num_shards), v_raw_y(num_shards), v_raw_z(num_shards);
    double sum_px = 0.0, sum_py = 0.0, sum_pz = 0.0;

    double base_speed = std::sqrt(std::max(1.0, 2.0 * dispersion_energy / m_total));

    for (int k = 0; k < num_shards; ++k) {
        double theta = dist_u(rng) * 3.14159265;
        double phi = dist_u(rng) * 3.14159265;
        double spd = base_speed * (0.5 + std::abs(dist_u(rng)));

        v_raw_x[k] = spd * std::sin(theta) * std::cos(phi);
        v_raw_y[k] = spd * std::cos(theta);
        v_raw_z[k] = spd * std::sin(theta) * std::sin(phi);

        sum_px += shard_masses[k] * v_raw_x[k];
        sum_py += shard_masses[k] * v_raw_y[k];
        sum_pz += shard_masses[k] * v_raw_z[k];
    }

    // Class 11 exact linear momentum balance shift: delta_v = - P_raw / m_total
    double shift_x = sum_px / m_total;
    double shift_y = sum_py / m_total;
    double shift_z = sum_pz / m_total;

    for (int k = 0; k < num_shards; ++k) {
        int s_idx = shard_indices[k];
        double* shard = state_tensor + (s_idx * NUM_FIELDS);

        shard[OFFSET_MASS] = shard_masses[k];
        shard[OFFSET_PX] = parent[OFFSET_PX];
        shard[OFFSET_PY] = parent[OFFSET_PY];
        shard[OFFSET_PZ] = parent[OFFSET_PZ];

        // Parent velocity + balanced dispersion velocity
        shard[OFFSET_VX] = parent_vx + (v_raw_x[k] - shift_x);
        shard[OFFSET_VY] = parent_vy + (v_raw_y[k] - shift_y);
        shard[OFFSET_VZ] = parent_vz + (v_raw_z[k] - shift_z);

        shard[OFFSET_RADIUS] = std::max(0.1, parent[OFFSET_RADIUS] * 0.25);
        shard[OFFSET_TYPE] = 5.0; // DEBRIS_FRAGMENT
        shard[OFFSET_CD] = 0.85;
        shard[OFFSET_AREA] = 0.15;
        shard[OFFSET_ACTIVE] = 1.0;
        shard[OFFSET_QW] = 1.0;
    }

    // Deactivate parent entity
    parent[OFFSET_ACTIVE] = 0.0;
}

} // extern "C"
