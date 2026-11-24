use classify;
use tracking;
use uniforms;

struct PickParams {
    ndc: vec2<f32>,
    slot: u32,
    _pad0: u32,
    mask_size: vec3<u32>,
    _pad1: u32,
}

struct PickResult {
    voxel: vec3<u32>,
    hit: u32,
}

@group(0) @binding(0) var<uniform> PICK: PickParams;
@group(0) @binding(1) var<storage, read_write> RESULT: PickResult;

@group(2) @binding(0) var<uniform> ENVIRONMENT: Environment;
@group(2) @binding(1) var BLUE_NOISE: texture_3d<f32>;

@group(3) @binding(0) var<uniform> HDRI_SETTINGS: HdriSettings;
@group(3) @binding(3) var LINE_EXTINCTION: texture_3d<f32>;
@group(3) @binding(4) var LINE_SAMPLER: sampler;

struct Material {
    extinction: vec3<f32>,
}

fn scattering_scale() -> f32 {
    return 1.0 / (1.0 - HDRI_SETTINGS.anisotropy);
}

fn to_material(sum: Classified) -> Material {
    let scattering = sum.scattering / (1.0 - HDRI_SETTINGS.anisotropy);
    return Material(sum.absorption + scattering);
}

fn unproject(v: vec3<f32>) -> vec3<f32> {
    let t = ENVIRONMENT.camera.projection_inverse * vec4<f32>(v, 1.0);
    return t.xyz / t.w;
}

fn intersectAABB(origin: vec3<f32>, direction: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    let invDir = 1.0 / direction;
    let t1 = (boxMin - origin) * invDir;
    let t2 = (boxMax - origin) * invDir;
    let tMinVec = min(t1, t2);
    let tMaxVec = max(t1, t2);
    let tMin = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    let tMax = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    return vec2<f32>(tMin, tMax);
}

fn classify_slot(uv: vec3<f32>, slot: u32) -> Classified {
    var sum = Classified();
    let crop = crop_attenuation(uv);
    if (crop <= 0.0 || slot >= CLASSIFY.count) { return sum; }

    switch (slot) {
        case 0u: { sum = add_fraction(sum, 0u, FRACTION_0, MASK_0, uv); }
        case 1u: { sum = add_fraction(sum, 1u, FRACTION_1, MASK_1, uv); }
        case 2u: { sum = add_fraction(sum, 2u, FRACTION_2, MASK_2, uv); }
        case 3u: { sum = add_fraction(sum, 3u, FRACTION_3, MASK_3, uv); }
        case 4u: { sum = add_fraction(sum, 4u, FRACTION_4, MASK_4, uv); }
        case 5u: { sum = add_fraction(sum, 5u, FRACTION_5, MASK_5, uv); }
        case 6u: { sum = add_fraction(sum, 6u, FRACTION_6, MASK_6, uv); }
        case 7u: { sum = add_fraction(sum, 7u, FRACTION_7, MASK_7, uv); }
        default: {}
    }

    sum.absorption *= crop;
    sum.scattering *= crop;
    return sum;
}

// Median Beer-Lambert collision optical depth: 1 - exp(-depth) = 0.5.
const HIT_DEPTH: f32 = 0.6931472;

@compute @workgroup_size(1)
fn main() {
    RESULT = PickResult(vec3<u32>(0u), 0u);

    let near = unproject(vec3<f32>(PICK.ndc, 0.0));
    let far = unproject(vec3<f32>(PICK.ndc, 1.0));

    let camera = TRANSFORM * ENVIRONMENT.camera.transform;
    let origin = (camera * vec4<f32>(near, 1.0)).xyz;
    let direction = (camera * vec4<f32>(normalize(far - near), 0.0)).xyz;

    let crop = CLASSIFY.crop;
    let hit = intersectAABB(
        origin,
        direction,
        max(0.5 + crop.min.xyz, vec3<f32>(0.0)),
        min(0.5 + crop.max.xyz, vec3<f32>(1.0)),
    );
    if (hit.x > hit.y) { return; }

    let entry = max(hit.x, 0.0);
    let exit = hit.y;
    if (entry >= exit) { return; }

    let dir_len = length(direction);
    if (dir_len <= 0.0) { return; }

    let step_t = 0.5 * voxel_step() / dir_len;
    var tracker = begin_tracking(origin, direction, entry, exit);

    var optical_depth = 0.0;
    var best_extinction = 0.0;
    var best_uv = vec3<f32>(0.0);
    var found = false;
    var hit_uv = vec3<f32>(0.0);

    while (tracking(tracker)) {
        let boundary = min(walk_exit(tracker.walk), tracker.exit);
        let majorant = block_majorant(tracker.walk.cell, scattering_scale());

        if (majorant > 0.0) {
            var t = tracker.t;
            while (t < boundary) {
                let dt = min(step_t, boundary - t);
                let uv = origin + direction * (t + 0.5 * dt);
                let ext = to_material(classify_slot(uv, PICK.slot)).extinction;
                let sigma = (ext.x + ext.y + ext.z) / 3.0;

                if (sigma > best_extinction) {
                    best_extinction = sigma;
                    best_uv = uv;
                }

                optical_depth += sigma * dt;
                if (optical_depth >= HIT_DEPTH) {
                    hit_uv = uv;
                    found = true;
                    break;
                }
                t += dt;
            }
            if (found) { break; }
        }

        tracker.t = max(tracker.t, boundary);
        tracker.walk = walk_next(tracker.walk);
    }

    if (!found && best_extinction > 0.0) {
        hit_uv = best_uv;
        found = true;
    }

    if (found) {
        let size = vec3<f32>(PICK.mask_size);
        let voxel = vec3<u32>(clamp(floor(hit_uv * size), vec3<f32>(0.0), size - 1.0));
        RESULT = PickResult(voxel, 1u);
    }
}
