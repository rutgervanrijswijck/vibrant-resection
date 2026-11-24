use egui::Rect;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    asset::{crop::CropSettings, volume_fraction::VolumeFractionBuffer},
    controller::camera::Camera,
};

/// Default spherical resection brush radius in millimeters.
pub const DEFAULT_BRUSH_RADIUS_MM: f32 = 8.0;

/// Builds a ray `(origin, direction)` in volume UV space `[0, 1]^3` for a click
/// at `click_px` (physical window pixels) inside `viewport` (physical pixels).
///
/// Matches the ray construction in `renderer/volume/render/trace.wgsl`, so the
/// ray parameter `t` in `origin + direction * t` is distance in millimeters.
pub fn ray_in_volume_uv(
    click_px: Vec2,
    viewport: Rect,
    camera: &Camera,
    volume_transform: Mat4,
) -> Option<(Vec3, Vec3)> {
    if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
        return None;
    }
    if !viewport.contains(egui::pos2(click_px.x, click_px.y)) {
        return None;
    }

    let ndc_x = ((click_px.x - viewport.min.x) / viewport.width()) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((click_px.y - viewport.min.y) / viewport.height()) * 2.0;

    let proj_inv = camera.projection().inverse();
    let unproject = |z: f32| {
        let t = proj_inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
        t.truncate() / t.w
    };

    let near = unproject(0.0);
    let far = unproject(1.0);
    let view_dir = (far - near).normalize_or_zero();
    if view_dir.length_squared() == 0.0 {
        return None;
    }

    let cam = volume_transform * camera.transform();
    let origin = cam.transform_point3(near);
    let direction = cam.transform_vector3(view_dir);
    Some((origin, direction))
}

/// Slab AABB intersection returning `(t_min, t_max)`. Hit when `t_min <= t_max`.
pub fn intersect_aabb(origin: Vec3, direction: Vec3, box_min: Vec3, box_max: Vec3) -> (f32, f32) {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for axis in 0..3 {
        let d = direction[axis];
        let o = origin[axis];
        if d.abs() < 1e-12 {
            if o < box_min[axis] || o > box_max[axis] {
                return (1.0, 0.0);
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (box_min[axis] - o) * inv_d;
            let mut t2 = (box_max[axis] - o) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
        }
    }

    (t_min, t_max)
}

/// Mirrors `spherical_crop` in `renderer/wgsl/attenuation.wgsl`.
fn spherical_crop_attenuation(crop: &CropSettings, uv: Vec3) -> f32 {
    let phi = crop.spherical.x;
    let theta = crop.spherical.y;
    let sin_theta = theta.sin();
    let normal = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos());
    let distance = normal.dot(uv - Vec3::splat(0.5));

    let edge0 = -crop.spherical.w;
    let edge1 = crop.spherical.w;
    let x = 0.5 - crop.spherical.z + crop.spherical.w - distance;
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Marches a ray `(origin, direction)` in volume UV space through `volume` and
/// returns `(t_mm, hit_uv)` for the first visible tissue surface hit.
pub fn raycast_volume(
    volume: &VolumeFractionBuffer,
    origin: Vec3,
    direction: Vec3,
    crop: &CropSettings,
) -> Option<(f32, Vec3)> {
    let box_min = (Vec3::splat(0.5) + crop.min.truncate()).max(Vec3::ZERO);
    let box_max = (Vec3::splat(0.5) + crop.max.truncate()).min(Vec3::ONE);
    let (t_min, t_max) = intersect_aabb(origin, direction, box_min, box_max);

    let entry = t_min.max(0.0);
    let exit = t_max;
    if entry > exit {
        return None;
    }

    let dir_len = direction.length();
    if dir_len <= 1e-8 {
        return None;
    }

    let size = volume.size();
    let size_f = size.as_vec3();
    let voxels = volume.voxels();
    let w = size.x as usize;
    let h = size.y as usize;

    // Step half a voxel at a time in UV space, expressed in ray parameter (mm).
    let step_mm = (0.5 / size.max_element().max(1) as f32) / dir_len;
    let mut tau = 0.0f32;
    let mut best: Option<(f32, f32, Vec3)> = None; // (density, t, uv)

    let mut t = entry;
    while t <= exit {
        let uv = origin + direction * t;
        let crop_att = spherical_crop_attenuation(crop, uv);
        if crop_att > 0.0 {
            let vx = ((uv.x * size_f.x).floor() as i32).clamp(0, size.x as i32 - 1) as usize;
            let vy = ((uv.y * size_f.y).floor() as i32).clamp(0, size.y as i32 - 1) as usize;
            let vz = ((uv.z * size_f.z).floor() as i32).clamp(0, size.z as i32 - 1) as usize;
            let raw = voxels[vx + w * (vy + h * vz)].to_f32();
            let density = volume.eval_density(raw) * crop_att;

            if density > 0.02 {
                tau += density * step_mm;
                if best.map_or(true, |(d, _, _)| density > d) {
                    best = Some((density, t, uv));
                }
                if tau >= 0.5 {
                    return Some((t, uv));
                }
            }
        }
        t += step_mm;
    }

    // Fallback for thin/semi-transparent tissue that didn't reach tau = 0.5.
    if tau >= 0.05 {
        best.map(|(_, t, uv)| (t, uv))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec3;

    #[test]
    fn aabb_hit_and_miss() {
        let (t0, t1) = intersect_aabb(Vec3::new(0.5, 0.5, -1.0), Vec3::Z, Vec3::ZERO, Vec3::ONE);
        assert!((t0 - 1.0).abs() < 1e-5);
        assert!((t1 - 2.0).abs() < 1e-5);

        let (m0, m1) = intersect_aabb(Vec3::new(2.0, 0.5, -1.0), Vec3::Z, Vec3::ZERO, Vec3::ONE);
        assert!(m0 > m1);
    }

    #[test]
    fn center_click_rays_along_camera_forward() {
        let mut camera = Camera::new();
        let viewport = Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(800.0, 600.0));
        camera.set_aspect(viewport.aspect_ratio());

        // Identity-like world->UV transform mapping [-100, 100]^3 mm to [0, 1]^3.
        let size = UVec3::splat(100);
        let affine = Mat4::from_translation(Vec3::splat(-100.0)) * Mat4::from_scale(Vec3::splat(2.0));
        let transform = Mat4::from_scale(Vec3::ONE / size.as_vec3()) * affine.inverse();

        let center = Vec2::new(100.0 + 400.0, 50.0 + 300.0);
        let (origin, direction) = ray_in_volume_uv(center, viewport, &camera, transform)
            .expect("center click should produce a ray");

        let (t0, t1) = intersect_aabb(origin, direction, Vec3::ZERO, Vec3::ONE);
        assert!(t0 <= t1 && t1 > 0.0, "center ray should intersect volume centered at origin");
    }
}
