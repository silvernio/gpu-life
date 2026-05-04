
struct Uniforms {
    aspect: f32,
    mouse: vec4<f32>,
    size: f32
}

struct Sim {
    colours: f32,
    beta: f32,
    rMax: f32,
    force: f32,
    friction: f32,
    dt: f32,
    cellSize: f32,
    cellAmt: f32,
    avoidance: f32,
    worldSize: f32,
    border: f32,
    vortex: f32,
}

struct Particle {
    pos: vec2f,
    vel: vec2f,
    colour: f32,
}

struct SortParticle {
    pos: vec2f,
    vel: vec2f,
    colour: f32,
    idx: vec2u
}

struct Force {
    total: vec2f,
    avoid: vec2f,
    div: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;

// Current Particle Buffer
@group(0) @binding(3) var<storage, read> input: array<Particle>;
// Next Particle Buffer
@group(0) @binding(4) var<storage, read_write> output: array<Particle>;

@group(0) @binding(5) var<storage, read> sorted: array<SortParticle>;
// The number of particles in each cell (counts generated in cell.wgsl)
@group(0) @binding(6) var<storage, read> counts: array<u32>;
// The index at which a cell's particles first appear
// Generated from the exclusive prefix sum of counts
@group(0) @binding(7) var<storage, read> prefixSumIndices: array<u32>;

fn hash3i(k: vec3<i32>) -> u32 {
    let offset: u32 = 0x80000000u;
    var x: u32 = (u32(k.x) + offset) * 0x9E3779B1u;
    var y: u32 = (u32(k.y) + offset) * 0x85EBCA6Bu;
    var z: u32 = (u32(k.z) + offset) * 0xC2B2AE35u;

    var h: u32 = x ^ y ^ z;

    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;

    return h;
}

fn force(r: f32, a: f32) -> vec2f {
    let beta = sim.beta;
	if (r < beta) {
		return vec2f(0, r / beta - 1);
	} else if (beta < r && r < 1) {
		return vec2f(a * (1 - abs(2 * r - 1 - beta) / (1 - beta)), 0);
	} else {
		return vec2f(0, 0);
	}
}

fn getCellForce(
    // Index of the particle (global_id.x)
    pi: u32,
    // Particle at input[global_id.x]
    p: Particle,
    // Localized cell hash
    cell: u32
) -> Force {
    var totalForceX = 0f;
    var totalForceY = 0f;

    var avoidForceX = 0f;
    var avoidForceY = 0f;

    let l = arrayLength(&input);
    let mrs = sim.rMax * sim.rMax;

    var div = 0f;

    let particlesThatCameBeforeThisCell = prefixSumIndices[cell]; // index.x
    let particlesInThisCell = counts[cell]; // index.y

    if (particlesInThisCell == 0u) {
        return Force(vec2f(0), vec2f(0), 0);
    }

    for (
        var i = particlesThatCameBeforeThisCell;
        i < particlesThatCameBeforeThisCell + particlesInThisCell;
        i++
    ) {
        let ip = sorted[i];
        if (ip.idx.y == pi) {
            continue;
        }

        let rx = ip.pos.x - p.pos.x;
        let ry = ip.pos.y - p.pos.y;

        let rs = rx * rx + ry * ry;
        if (rs > 0 && rs < mrs) {
            let r = sqrt(rs);
            let f = force(r / sim.rMax, matrix[u32(p.colour) * u32(sim.colours) + u32(ip.colour)]);
            totalForceX += rx / r * f.x;
            totalForceY += ry / r * f.x;

            avoidForceX += rx / r * f.y;
            avoidForceY += ry / r * f.y;

            div += max(0, -f.y) / 10 * sim.avoidance;
        } else if (rs == 0) {
            avoidForceX += (f32(pi)*20293.302 % 4932.329) / 100000;
        }
    }

    return Force(vec2f(totalForceX, totalForceY), vec2f(avoidForceX, avoidForceY), div);
}

fn getForce(pi: u32) -> vec2f {
    let p= input[pi];

    var totalForce = vec2f(0, 0);

    var avoidForce = vec2f(0, 0);

    var div = 0f;

    let gridPos = vec2i(floor(p.pos / sim.cellSize));

    let cxmin = i32(floor((p.pos.x - sim.rMax) / sim.cellSize));
    let cymin = i32(floor((p.pos.y - sim.rMax) / sim.cellSize));
    let cxmax = i32(floor((p.pos.x + sim.rMax) / sim.cellSize));
    let cymax = i32(floor((p.pos.y + sim.rMax) / sim.cellSize));

    for (var cx = cxmin; cx <= cxmax; cx++) {
        for (var cy = cymin; cy <= cymax; cy++) {
            let c = hash3i(vec3i(cx, cy, 0)) % u32(sim.cellAmt);
            let force = getCellForce(pi, p, c);
            totalForce += force.total;
            avoidForce += force.avoid;
            div += force.div;
        }
    }

    totalForce *= sim.rMax * sim.force / (1 + div);
    return totalForce + avoidForce;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&output)) {
        return;
    }
    var p  = input[global_id.x];

    let mx = uniforms.mouse.x;

    let force = getForce(global_id.x);

    p.vel.x *= sim.friction;
    p.vel.y *= sim.friction;

    p.vel.x += force.x * sim.dt;
    p.vel.y += force.y * sim.dt;

    let d = sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y) / sim.worldSize;

    if (sim.vortex > 0) {
        p.vel.x -= p.pos.x * sim.dt * 5 * cos(d * 30 * p.pos.x * p.pos.y / matrix[u32(p.colour) * 10]) * sin(p.pos.x * 10 * matrix[u32(p.colour)]);
        p.vel.y -= p.pos.y * sim.dt * 5 * cos(d * 30 * p.pos.y * p.pos.x / matrix[u32(p.colour) * 10]) * cos(p.pos.y * 10 * matrix[u32(p.colour)]);

        p.vel.x -= p.pos.y * sim.dt * 5 * pow(d / 0.5, 2) * sin(d * 10 + matrix[u32(p.colour) + 10]);
        p.vel.y += p.pos.x * sim.dt * 5 * pow(d / 0.5, 2) * sin(d * 10 + matrix[u32(p.colour) + 10]);
    }

    if (uniforms.mouse.z != 0) {
        let dx = uniforms.mouse.x - p.pos.x;
        let dy = uniforms.mouse.y - p.pos.y;

        let d = sqrt(dx * dx + dy * dy);
        if (d < sim.rMax) {
            if (uniforms.mouse.z == 1) {
                p.vel.x += dx * 3;
                p.vel.y += dy * 3;
            } else {
                p.vel.x -=  dx * 3;
                p.vel.y -= dy * 3;
            }
        }
    }

    if (sim.border > 0 && d > 0.9) {
        let f = (d - 0.9) * 10;
        p.vel.x -= p.pos.x * f;
        p.vel.y -= p.pos.y * f;
    }

    p.pos.x += p.vel.x * sim.dt;
    p.pos.y += p.vel.y * sim.dt;

    output[global_id.x] = p;
}