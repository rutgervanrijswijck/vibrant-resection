use std::{
    any::type_name,
    array,
    sync::atomic::{AtomicU64, Ordering},
};

use bytemuck::{bytes_of, checked::cast_slice, Pod, Zeroable};
use glam::{Mat4, UVec3, Vec3};
use half::f16;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    wgt::TextureDataOrder,
    *,
};

use crate::{
    asset::{
        colormap::{Colormap, ColormapSelection},
        material::{Material, MaterialNode, MaterialPreset},
    },
    file::{Bins, VolumeFile},
    gpu::Gpu,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// (normalized intensity, height in [0, 1]) per bin.
    points: Vec<(f32, f32)>,
}

impl Histogram {
    /// Between linear and log: `ln(1 + d/k)` with `k` a thirtieth of the
    /// reference, so sparse bins stay visible while peaks keep their contrast.
    /// The reference is the 99th-percentile bin; the background spike clips.
    pub fn from_bins(bins: &Bins) -> Self {
        let density = &bins.density;
        let mut sorted = density.clone();
        sorted.sort_unstable_by(f32::total_cmp);
        let reference = sorted
            .get((sorted.len().max(1) - 1) * 99 / 100)
            .copied()
            .unwrap_or(0.0)
            .max(f32::MIN_POSITIVE);

        let k = reference / 30.0;
        let scale = 1.0 / (reference / k).ln_1p();

        Self {
            points: density
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    (
                        bins.first + i as f32 * bins.step,
                        ((d / k).ln_1p() * scale).min(1.0),
                    )
                })
                .collect(),
        }
    }

    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }
}

/// Stable across the volume's lifetime, unlike its Vec index. Used to give UI
/// widgets (e.g. the transfer function editor) identity that doesn't collide
/// across volumes or shift when volumes are added/removed.
static NEXT_VOLUME_ID: AtomicU64 = AtomicU64::new(0);

fn next_volume_id() -> u64 {
    NEXT_VOLUME_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq)]
pub struct VolumeFractionSettings {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub mask: usize,
    pub inverted: bool,
    pub masked: bool,
    pub use_colormap: bool,
    pub colormap: ColormapSelection,
    pub opacity: f32,
    pub histogram: Histogram,
    pub nodes: Vec<MaterialNode>,
}

impl VolumeFractionSettings {
    fn to_buffer(&self) -> VolumeFractionSettingsBuffer {
        VolumeFractionSettingsBuffer {
            inverted: self.inverted as u32,
            masked: self.masked as u32,
            use_colormap: self.use_colormap as u32,
            colormap: self.colormap as u32,
            opacity: self.opacity,
            _pad: [0.0; 3],
            nodes: array::from_fn(|index| {
                self.nodes
                    .get(index)
                    .map(|node| node.into())
                    .unwrap_or(MaterialNodeBuffer {
                        position: 0.0,
                        absorption: [0.0, 0.0, 0.0],
                        scattering: [0.0, 0.0, 0.0],
                        ior: 0.0,
                    })
            }),
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct MaterialNodeBuffer {
    absorption: [f32; 3],
    position: f32,
    scattering: [f32; 3],
    ior: f32,
}

impl MaterialNodeBuffer {
    pub const MAX_NODES: usize = 8;
}

impl From<&MaterialNode> for MaterialNodeBuffer {
    fn from(value: &MaterialNode) -> Self {
        MaterialNodeBuffer {
            absorption: value.material.absorption,
            position: value.position,
            scattering: value.material.scattering,
            ior: value.material.ior,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct VolumeFractionSettingsBuffer {
    inverted: u32,
    masked: u32,
    use_colormap: u32,
    colormap: u32,
    opacity: f32,
    _pad: [f32; 3],
    nodes: [MaterialNodeBuffer; MaterialNodeBuffer::MAX_NODES],
}

pub struct VolumeFractionBuffer {
    texture: Texture,
    view: TextureView,
    size: UVec3,

    original_voxels: Vec<f16>,
    voxels: Vec<f16>,
    removed: Vec<u8>,
    removed_count: usize,

    transform: Mat4,
    transform_buffer: Buffer,

    settings: VolumeFractionSettings,
    settings_buffer: Buffer,

    binding: BindGroup,
}

impl VolumeFractionBuffer {
    pub fn new(gpu: &Gpu, file: &VolumeFile, colormap: &Colormap) -> Self {
        let label = Some(type_name::<Self>());

        let size = Extent3d {
            width: file.size().x,
            height: file.size().y,
            depth_or_array_layers: file.size().z,
        };

        let texture = gpu.device().create_texture_with_data(
            gpu.queue(),
            &TextureDescriptor {
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::R16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::LayerMajor,
            cast_slice(file.data()),
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        let sampler = gpu.device().create_sampler(&SamplerDescriptor {
            label,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let transform = file.transform();

        let transform_buffer = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label,
            contents: bytes_of(&transform),
            usage: BufferUsages::UNIFORM,
        });

        let settings = VolumeFractionSettings {
            id: next_volume_id(),
            name: file.name().to_string(),
            visible: true,
            mask: 0,
            inverted: false,
            masked: true,
            use_colormap: false,
            colormap: ColormapSelection::Viridis,
            opacity: 1.0,
            histogram: Histogram::from_bins(file.histogram()),
            nodes: vec![
                MaterialNode {
                    id: 0,
                    selected: false,
                    position: 0.0,
                    preset: MaterialPreset::Air,
                    material: MaterialPreset::Air.material().unwrap(),
                },
                MaterialNode {
                    id: 1,
                    selected: true,
                    position: 1.0,
                    preset: MaterialPreset::Custom,
                    material: Material {
                        absorption: [1.0, 1.0, 1.0],
                        scattering: [1.0, 1.0, 1.0],
                        ior: 1.5,
                    },
                },
            ],
        };

        let settings_buffer = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label,
            contents: bytes_of(&settings.to_buffer()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let binding = gpu.device().create_bind_group(&BindGroupDescriptor {
            label,
            layout: &Self::layout(gpu),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: transform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: settings_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &colormap
                            .texture()
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let original_voxels = file.data().to_vec();
        let voxels = original_voxels.clone();
        let removed = vec![0u8; original_voxels.len()];

        Self {
            texture,
            view,
            size: file.size(),
            original_voxels,
            voxels,
            removed,
            removed_count: 0,
            transform,
            transform_buffer,
            settings,
            settings_buffer,
            binding,
        }
    }

    pub fn voxels(&self) -> &[f16] {
        &self.voxels
    }

    pub fn removed_count(&self) -> usize {
        self.removed_count
    }

    /// Evaluates the transfer-function extinction density at a normalized voxel
    /// value, mirroring `material.wgsl` + `lut.wgsl` (`brightness(a + s)`).
    pub fn eval_density(&self, raw: f32) -> f32 {
        let value = if self.settings.inverted && raw != 0.0 {
            1.0 - raw
        } else {
            raw
        }
        .clamp(0.0, 1.0);

        let nodes = &self.settings.nodes;
        if nodes.is_empty() {
            return 0.0;
        }

        let (absorption, scattering) = if nodes.len() == 1 {
            (nodes[0].material.absorption, nodes[0].material.scattering)
        } else if let Some(index) = nodes.iter().position(|n| n.position >= value) {
            if index == 0 {
                (nodes[0].material.absorption, nodes[0].material.scattering)
            } else {
                let a = &nodes[index - 1];
                let b = &nodes[index];
                let span = (b.position - a.position).max(1e-5);
                let t = ((value - a.position) / span).clamp(0.0, 1.0);
                let lerp3 = |u: [f32; 3], v: [f32; 3]| {
                    [
                        u[0] + (v[0] - u[0]) * t,
                        u[1] + (v[1] - u[1]) * t,
                        u[2] + (v[2] - u[2]) * t,
                    ]
                };
                (
                    lerp3(a.material.absorption, b.material.absorption),
                    lerp3(a.material.scattering, b.material.scattering),
                )
            }
        } else {
            let last = nodes.last().unwrap();
            (last.material.absorption, last.material.scattering)
        };

        let opacity = self.settings.opacity.clamp(0.0, 1.0);
        let ext = [
            (absorption[0] + scattering[0]) * opacity,
            (absorption[1] + scattering[1]) * opacity,
            (absorption[2] + scattering[2]) * opacity,
        ];
        0.2126 * ext[0] + 0.7152 * ext[1] + 0.0722 * ext[2]
    }

    /// Marks all voxels within `radius_mm` of `center_uv` (in world mm) as
    /// removed, zeros them in the GPU texture, and returns the count of newly
    /// removed voxels.
    pub fn resect_sphere(&mut self, gpu: &Gpu, center_uv: Vec3, radius_mm: f32) -> usize {
        let newly_removed = self.resect_sphere_cpu(center_uv, radius_mm);
        if let Some((count, z_min, z_max)) = newly_removed {
            self.upload_z_slices(gpu, z_min, z_max);
            count
        } else {
            0
        }
    }

    /// CPU half of [`Self::resect_sphere`]; returns `(newly_removed, z_min, z_max)`.
    fn resect_sphere_cpu(&mut self, center_uv: Vec3, radius_mm: f32) -> Option<(usize, u32, u32)> {
        if radius_mm <= 0.0 {
            return None;
        }

        let size_f = self.size.as_vec3();
        let uv_to_world = self.transform.inverse();
        let center_world = uv_to_world.transform_point3(center_uv);

        // Exact axis-aligned voxel-space half-extents of a world-space sphere.
        let world_to_voxel = Mat4::from_scale(size_f) * self.transform;
        let extent = Vec3::new(
            world_to_voxel.row(0).truncate().length() * radius_mm,
            world_to_voxel.row(1).truncate().length() * radius_mm,
            world_to_voxel.row(2).truncate().length() * radius_mm,
        );

        let center_voxel = center_uv * size_f;
        let min_v = (center_voxel - extent - Vec3::ONE)
            .floor()
            .max(Vec3::ZERO)
            .as_uvec3();
        let max_v = (center_voxel + extent + Vec3::ONE)
            .ceil()
            .min(size_f - Vec3::ONE)
            .as_uvec3();

        if min_v.x > max_v.x || min_v.y > max_v.y || min_v.z > max_v.z {
            return None;
        }

        let r2 = radius_mm * radius_mm;
        let w = self.size.x as usize;
        let h = self.size.y as usize;
        let mut count = 0usize;

        for z in min_v.z..=max_v.z {
            for y in min_v.y..=max_v.y {
                let row_base = w * (y as usize + h * z as usize);
                for x in min_v.x..=max_v.x {
                    let uv = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) / size_f;
                    let world = uv_to_world.transform_point3(uv);
                    if world.distance_squared(center_world) <= r2 {
                        let idx = row_base + x as usize;
                        if self.removed[idx] == 0 {
                            self.removed[idx] = 1;
                            self.voxels[idx] = f16::ZERO;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            self.removed_count += count;
            Some((count, min_v.z, max_v.z))
        } else {
            None
        }
    }

    /// Restores all resected voxels to their original intensities.
    pub fn reset_resection(&mut self, gpu: &Gpu) -> bool {
        if self.removed_count == 0 {
            return false;
        }
        self.voxels.copy_from_slice(&self.original_voxels);
        self.removed.fill(0);
        self.removed_count = 0;
        self.upload_z_slices(gpu, 0, self.size.z - 1);
        true
    }

    fn upload_z_slices(&self, gpu: &Gpu, z_min: u32, z_max: u32) {
        let slice_len = (self.size.x * self.size.y) as usize;
        let start = z_min as usize * slice_len;
        let end = (z_max as usize + 1) * slice_len;
        let slice_bytes: &[u8] = cast_slice(&self.voxels[start..end]);

        gpu.queue().write_texture(
            TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: z_min,
                },
                aspect: TextureAspect::All,
            },
            slice_bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size.x * 2),
                rows_per_image: Some(self.size.y),
            },
            Extent3d {
                width: self.size.x,
                height: self.size.y,
                depth_or_array_layers: z_max - z_min + 1,
            },
        );
    }

    pub fn size(&self) -> UVec3 {
        self.size
    }

    pub fn view(&self) -> &TextureView {
        &self.view
    }

    pub fn binding(&self) -> &BindGroup {
        &self.binding
    }

    pub fn transform(&self) -> Mat4 {
        self.transform
    }

    pub fn settings(&self) -> &VolumeFractionSettings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut VolumeFractionSettings {
        &mut self.settings
    }

    pub fn update_settings(&self, gpu: &Gpu) {
        gpu.queue().write_buffer(
            &self.settings_buffer,
            0,
            bytes_of(&self.settings.to_buffer()),
        );
    }

    pub fn layout(gpu: &Gpu) -> BindGroupLayout {
        let visibility = ShaderStages::FRAGMENT | ShaderStages::COMPUTE;

        gpu.device()
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(type_name::<Self>()),
                entries: &[
                    // Texture
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Transform
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Settings
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Colormap
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            })
    }
}

impl Drop for VolumeFractionBuffer {
    fn drop(&mut self) {
        self.texture.destroy();
        self.transform_buffer.destroy();
    }
}
