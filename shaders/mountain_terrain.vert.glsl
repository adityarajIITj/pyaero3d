#version 150
// PyAero3D - GPU Mountain Terrain Vertex Shader (Panda3D ShaderTerrainMesh)
// Deforms vertices based on 16-bit heightfield with Continuous Level of Detail (CLOD) morphing

in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

uniform struct {
    sampler2D data_texture;
    sampler2D heightfield;
    int view_index;
    int terrain_size;
    int chunk_size;
} ShaderTerrainMesh;

out vec2 terrain_uv;
out vec3 world_pos;
out float elevation_val;

void main() {
    // Terrain chunk data layout: x = chunk_x, y = chunk_y, z = size, w = clod_factor
    vec4 terrain_data = texelFetch(ShaderTerrainMesh.data_texture,
        ivec2(gl_InstanceID, ShaderTerrainMesh.view_index), 0);

    vec3 chunk_position = p3d_Vertex.xyz;

    // Continuous Level of Detail (CLOD) morphing to prevent geometric vertex popping
    float clod_factor = smoothstep(0.0, 1.0, terrain_data.w);
    vec2 clamped_coord = clamp(chunk_position.xy, clod_factor, 1.0 - clod_factor);
    chunk_position.xy = mix(chunk_position.xy, clamped_coord, clod_factor);

    // Compute absolute terrain UV coordinates
    terrain_uv = (chunk_position.xy * terrain_data.z + terrain_data.xy) / float(ShaderTerrainMesh.terrain_size);

    // Sample elevation from 16-bit heightfield texture
    float height = texture(ShaderTerrainMesh.heightfield, terrain_uv).r;

    // Transform position into world space
    vec4 final_pos = vec4(
        (chunk_position.x * terrain_data.z + terrain_data.xy.x),
        (chunk_position.y * terrain_data.z + terrain_data.xy.y),
        height,
        1.0
    );

    vec4 p3d_world_pos = p3d_ModelMatrix * final_pos;
    world_pos = p3d_world_pos.xyz;
    elevation_val = height;

    gl_Position = p3d_ModelViewProjectionMatrix * final_pos;
}
