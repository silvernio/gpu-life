struct SortParticle {
    pos: vec2f,
    vel: vec2f,
    colour: f32,
    idx: vec2u
}

// Cell Buffer:
@group(0) @binding(0) var<storage, read> input: array<SortParticle>;
// Sorted Buffer:
@group(0) @binding(1) var<storage, read_write> output: array<SortParticle>;
// Result of our prefixSum
@group(0) @binding(2) var<storage, read_write> prefixSumIndices: array<atomic<u32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&input)) {
        return;
    }
    // Get the current sortParticle
    let value = input[global_id.x];
    // Write at an offset of 1 so result in sim.wgsl is true exclusive prefix sum
    let targetIndex = atomicAdd(&prefixSumIndices[value.idx.x + 1u], 1u);
    output[targetIndex] = value;
}