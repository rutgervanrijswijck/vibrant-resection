use std::any::type_name;

use bytemuck::{bytes_of, Pod, Zeroable};
use glam::{UVec3, Vec2};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    RenderPassDescriptor, RenderPipeline, ShaderStages, TextureSampleType, TextureViewDimension,
};

use crate::{
    asset::{
        environment::Environment,
        hdri::HdriBuffer,
        radiance::GaussianRadianceBuffer,
        texture::{MipTexture3D, R32Float},
        volume::PhysicalVolume,
        Asset,
    },
    controller::Controller,
    gpu::{readback::Readback, Gpu, Lazy},
    surface::{color::ColorBuffer, Frame},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PickParams {
    ndc: [f32; 2],
    slot: u32,
    _pad0: u32,
    mask_size: [u32; 3],
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PickResult {
    voxel: [u32; 3],
    hit: u32,
}

pub struct GaussianVolumeRenderer {
    pipeline: Lazy<RenderPipeline>,
    /// Per microfacet and mirror Fresnel.
    traced: [Lazy<RenderPipeline>; 2],
    pick: Lazy<ComputePipeline>,
    pick_params: Buffer,
    pick_result: Buffer,
    pick_binding: BindGroup,
    pick_readback: Readback<(u64, UVec3)>,
    // Group 3: HDRI material settings + the combined-mode line-depth target.
    // Built per dispatch since the line-depth view lives on the (resizable) Frame.
    material_layout: BindGroupLayout,
    classified_layout: BindGroupLayout,
}

impl GaussianVolumeRenderer {
    pub fn new(gpu: &Gpu) -> Self {
        let material_layout = Self::material_layout(gpu);
        let classified_layout = PhysicalVolume::layout_classified(gpu);

        let env_layout = Environment::layout(gpu);
        let layout = gpu.pipeline_layout(&[
            &GaussianRadianceBuffer::layout(gpu),
            &classified_layout,
            &env_layout,
            &material_layout,
        ]);

        let pick_layout = gpu
            .device()
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("VolumePick"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pick_pipeline_layout = gpu.pipeline_layout(&[
            &pick_layout,
            &classified_layout,
            &env_layout,
            &material_layout,
        ]);

        let pick_params = gpu.device().create_buffer(&BufferDescriptor {
            label: Some("VolumePick::Params"),
            size: size_of::<PickParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pick_result = gpu.device().create_buffer(&BufferDescriptor {
            label: Some("VolumePick::Result"),
            size: size_of::<PickResult>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pick_binding = gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some("VolumePick"),
            layout: &pick_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pick_params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pick_result.as_entire_binding(),
                },
            ],
        });

        let pick = gpu.lazy("VolumePick", move |gpu| {
            gpu.compute(
                "VolumePick",
                &pick_pipeline_layout,
                &gpu.shader("renderer/volume/render/pick.wgsl"),
            )
        });

        let pipeline = |label: &'static str, traced: bool, mirror: bool| {
            let layout = layout.clone();
            gpu.lazy(label, move |gpu| {
                gpu.quad(
                    label,
                    &layout,
                    ColorBuffer::target_premultiplied(),
                    &gpu.shader_with(
                        "renderer/volume/render/trace.wgsl",
                        &[("TRACED", &traced), ("MIRROR_FRESNEL", &mirror)],
                    ),
                )
            })
        };

        Self {
            pipeline: pipeline("GaussianVolumeTrace", false, false),
            traced: [
                pipeline("GaussianVolumeTrace::Traced", true, false),
                pipeline("GaussianVolumeTrace::TracedMirror", true, true),
            ],
            pick,
            pick_params,
            pick_result,
            pick_binding,
            pick_readback: Readback::new(),
            material_layout,
            classified_layout,
        }
    }

    /// Group 3 for `trace.wgsl`: the `HdriSettings` uniform, the line-depth
    /// target, and for the ray traced mode the environment map and line density.
    fn material_layout(gpu: &Gpu) -> BindGroupLayout {
        gpu.device()
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(type_name::<Self>()),
                entries: &[
                    vec![
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                    MipTexture3D::<R32Float>::layout_entries(3),
                ]
                .concat(),
            })
    }

    pub fn dispatch(
        &self,
        gpu: &Gpu,
        cmd: &mut CommandEncoder,
        asset: &Asset,
        controller: &Controller,
        frame: &Frame,
    ) {
        let (Some(volume), Some(radiance)) = (&asset.physical_volume, &asset.radiance) else {
            return;
        };

        let classified = volume.binding_classified(
            gpu,
            &self.classified_layout,
            &asset.visible_volumes(),
            &asset.masks,
        );

        let material = self.material_binding(gpu, &asset.hdri, volume, frame);

        let mut pass = cmd.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(frame.color().attachment())],
            ..Default::default()
        });

        let viewport = controller.viewport();
        pass.set_viewport(
            viewport.min.x,
            viewport.min.y,
            viewport.width(),
            viewport.height(),
            0.0,
            1.0,
        );

        let settings = controller.settings();
        if settings.ray_traced {
            pass.set_pipeline(&self.traced[settings.mirror_fresnel as usize]);
        } else {
            pass.set_pipeline(&self.pipeline);
        }

        pass.set_bind_group(0, radiance.binding(), &[]);
        pass.set_bind_group(1, &classified, &[]);
        pass.set_bind_group(2, asset.environment.binding(), &[]);
        pass.set_bind_group(3, &material, &[]);
        pass.draw(0..4, 0..1);
    }

    /// Dispatches a 1-thread GPU ray trace for a resection click at `ndc`,
    /// returning the target volume's id if a trace was queued.
    pub fn pick(
        &self,
        gpu: &Gpu,
        cmd: &mut CommandEncoder,
        asset: &Asset,
        controller: &Controller,
        frame: &Frame,
        ndc: Vec2,
    ) -> Option<u64> {
        let volume = asset.physical_volume.as_ref()?;
        let target = asset.volumes.get(controller.resection().volume())?;
        let target_id = target.settings().id;
        let visible = asset.visible_volumes();
        let slot = visible.iter().position(|v| v.settings().id == target_id)?;

        let params = PickParams {
            ndc: ndc.to_array(),
            slot: slot as u32,
            _pad0: 0,
            mask_size: target.size().to_array(),
            _pad1: 0,
        };
        gpu.queue()
            .write_buffer(&self.pick_params, 0, bytes_of(&params));

        let classified =
            volume.binding_classified(gpu, &self.classified_layout, &visible, &asset.masks);
        let material = self.material_binding(gpu, &asset.hdri, volume, frame);

        let mut pass = cmd.begin_compute_pass(&ComputePassDescriptor {
            label: Some("VolumePick"),
            ..Default::default()
        });
        pass.set_pipeline(&self.pick);
        pass.set_bind_group(0, &self.pick_binding, &[]);
        pass.set_bind_group(1, &classified, &[]);
        pass.set_bind_group(2, asset.environment.binding(), &[]);
        pass.set_bind_group(3, &material, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        Some(target_id)
    }

    /// Reads back the result of the last [`Self::pick`] dispatch.
    pub fn read_pick(&self, gpu: &Gpu, volume_id: u64) {
        let gpu = gpu.clone();
        let buffer = self.pick_result.clone();
        self.pick_readback.refresh(async move {
            let result = gpu.read_buffer::<PickResult>(&buffer).await?;
            let hit = *result.first()?;
            (hit.hit != 0).then(|| (volume_id, UVec3::from(hit.voxel)))
        });
    }

    pub fn take_pick(&self) -> Option<(u64, UVec3)> {
        self.pick_readback.take()
    }

    pub fn pick_pending(&self) -> bool {
        self.pick_readback.pending() || self.pick_readback.get().is_some()
    }

    fn material_binding(
        &self,
        gpu: &Gpu,
        hdri: &HdriBuffer,
        volume: &PhysicalVolume,
        frame: &Frame,
    ) -> BindGroup {
        gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some(type_name::<Self>()),
            layout: &self.material_layout,
            entries: &[
                vec![
                    BindGroupEntry {
                        binding: 0,
                        resource: hdri.settings_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(frame.line_depth().view()),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(hdri.map()),
                    },
                ],
                volume.line_extinction().binding_entries(3),
            ]
            .concat(),
        })
    }
}
