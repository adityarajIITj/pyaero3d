#version 150
// PyAero3D - GPU Mountain Terrain Fragment Shader
// Features: Analytical gradient normals, slope-dependent procedural texturing,
// directional solar lighting, and exponential atmospheric scattering fog.

in vec2 terrain_uv;
in vec3 world_pos;
in float elevation_val;
out vec4 frag_color;

uniform struct {
    sampler2D data_texture;
    sampler2D heightfield;
    int view_index;
    int terrain_size;
    int chunk_size;
} ShaderTerrainMesh;

uniform vec3 wspos_camera;

// Computes analytical surface normal directly from heightfield gradient
vec3 compute_surface_normal() {
    float texel = 1.0 / float(ShaderTerrainMesh.terrain_size);
    float h_l = texture(ShaderTerrainMesh.heightfield, terrain_uv + vec2(-texel, 0.0)).r;
    float h_r = texture(ShaderTerrainMesh.heightfield, terrain_uv + vec2(texel, 0.0)).r;
    float h_d = texture(ShaderTerrainMesh.heightfield, terrain_uv + vec2(0.0, -texel)).r;
    float h_u = texture(ShaderTerrainMesh.heightfield, terrain_uv + vec2(0.0, texel)).r;

    // Normal gradient
    vec3 normal = normalize(vec3((h_l - h_r) * 12.0, (h_d - h_u) * 12.0, 2.0 * texel * float(ShaderTerrainMesh.terrain_size)));
    return normal;
}

void main() {
    vec3 N = compute_surface_normal();
    vec3 sun_dir = normalize(vec3(0.4, 0.3, 0.85)); // Sun angle

    // Slope calculation: dot(N, up_vector)
    float slope = N.z; // 1.0 = flat plateau/valley, 0.0 = sheer vertical cliff

    // 1. Procedural Alpine Multi-Layer Texturing
    vec3 grass_color = vec3(0.18, 0.32, 0.22); // Valley alpine grass (#2E5339)
    vec3 rock_color  = vec3(0.29, 0.31, 0.32); // Steep granite slate (#4A4E51)
    vec3 snow_color  = vec3(0.92, 0.94, 0.98); // High-altitude snow (#E8F0FE)

    // Blend rock on steep slopes (slope < 0.70)
    float rock_factor = 1.0 - smoothstep(0.45, 0.75, slope);
    vec3 base_surface = mix(grass_color, rock_color, rock_factor);

    // Blend snow at high elevations (> 0.55 normalized elevation)
    float snow_factor = smoothstep(0.50, 0.75, elevation_val) * smoothstep(0.30, 0.70, slope);
    base_surface = mix(base_surface, snow_color, snow_factor);

    // 2. Directional Solar Diffuse Lighting + Ambient
    float n_dot_l = max(dot(N, sun_dir), 0.0);
    vec3 ambient = vec3(0.20, 0.25, 0.30);
    vec3 lit_color = base_surface * (ambient + vec3(0.85, 0.80, 0.75) * n_dot_l);

    // 3. Exponential Rayleigh/Mie Atmospheric Scattering Distance Fog
    float dist = length(world_pos - wspos_camera);
    float fog_density = 0.00015;
    float fog_factor = 1.0 - exp(-dist * fog_density);
    vec3 fog_color = vec3(0.65, 0.75, 0.88); // Sky horizon blue

    vec3 final_rgb = mix(lit_color, fog_color, clamp(fog_factor, 0.0, 0.90));
    frag_color = vec4(final_rgb, 1.0);
}
