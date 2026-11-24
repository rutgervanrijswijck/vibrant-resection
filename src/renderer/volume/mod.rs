pub mod render;
pub mod transfer;

use glam::{UVec3, Vec2};
use wgpu::CommandEncoder;

use crate::{
    asset::Asset,
    controller::Controller,
    gpu::Gpu,
    renderer::volume::{render::GaussianVolumeRenderer, transfer::VolumeTransferPipeline},
    surface::Frame,
};

pub struct VolumeRenderer {
    transfer: VolumeTransferPipeline,
    render: GaussianVolumeRenderer,
}

impl VolumeRenderer {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            transfer: VolumeTransferPipeline::new(gpu),
            render: GaussianVolumeRenderer::new(gpu),
        }
    }

    /// After any fraction, mask or crop edit.
    pub fn transfer(&self, gpu: &Gpu, cmd: &mut CommandEncoder, asset: &Asset) {
        self.transfer.dispatch(gpu, cmd, asset);
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        cmd: &mut CommandEncoder,
        controller: &Controller,
        asset: &Asset,
        frame: &Frame,
    ) {
        // In a lines-only scene the PhysicalVolume holds nothing but deposited
        // line density (used to build the cascade); tracing it would just wrap
        // every line in haze.
        if asset.volumes.is_empty() {
            return;
        }

        self.render.dispatch(gpu, cmd, asset, controller, frame);
    }

    pub fn pick(
        &self,
        gpu: &Gpu,
        cmd: &mut CommandEncoder,
        controller: &Controller,
        asset: &Asset,
        frame: &Frame,
        ndc: Vec2,
    ) -> Option<u64> {
        self.render.pick(gpu, cmd, asset, controller, frame, ndc)
    }

    pub fn read_pick(&self, gpu: &Gpu, volume_id: u64) {
        self.render.read_pick(gpu, volume_id);
    }

    pub fn take_pick(&self) -> Option<(u64, UVec3)> {
        self.render.take_pick()
    }

    pub fn pick_pending(&self) -> bool {
        self.render.pick_pending()
    }
}
