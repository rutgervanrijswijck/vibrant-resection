# Context
This project is a fork of **Vibrant**, a real-time 3D medical visualization tool focused on physically based volume rendering of brain scans (NIfTI) and tractography (`.tck` / `.tsf`). It is built in Rust using `wgpu`, `winit`, and `egui` / `egui_dock`, targeting both native desktop and WebGPU (`wasm32`).

# Project Structure
- `src/app.rs`: `winit` application loop (`App`), window events, redraw scheduling, and input routing to `egui` and `Controller`.
- `src/file/`: Async file loading (`FileStage`) and parsers for NIfTI volumes (`volume.rs`, normalized to `f16` with world↔UV transforms), tractograms/scalars (`line.rs`), and `.exr` HDRIs (`hdri.rs`).
- `src/asset/`: GPU-backed scene resources managed by `Asset` (`mod.rs`):
  - `volume_fraction.rs`: Scalar volume textures (`R16Float`), histograms, and per-volume transfer-function/material settings.
  - `volume_mask.rs`: 3D masks (auto-detected when a filename contains `"mask"`), supporting binary and SDF modes.
  - `volume.rs`: `PhysicalVolume` holding the shared classified medium, LUTs, majorant/coarse empty-space grids, and baked extinction.
  - `crop.rs`: Box and spherical clipping planes (`CropBuffer`).
  - `line/` & `tractography.rs`: Streamline buffers and voxel acceleration structures.
  - `radiance.rs`, `environment.rs`, `hdri/`: Camera/environment uniforms and 3D radiance cascade buffers.
- `src/controller/`: `Controller` (`mod.rs`), orbit/pan/zoom `Camera` (`camera.rs`), `Light` (`light.rs`), `Settings` (`settings.rs`), input state (`state.rs`), and `egui_dock` UI panels in `widgets/` (`volumes.rs`, `masks.rs`, `crop.rs`, `tractography.rs`, `rendering.rs`, `controls.rs`).
- `src/renderer/`: `Renderer` (`mod.rs`) and WGSL shaders (`wgsl/`):
  - `volume/`: Transfer-function/majorant/extinction compute passes (`transfer/`) and delta/ratio-tracking volume raymarcher (`render/trace.wgsl`).
  - `lighting/`: 3D Gaussian radiance cascade passes.
  - `line/`: Voxelized tractography population, deposit, and ray tracing.
  - `accumulate/`, `present/`, `ui.rs`: Progressive frame accumulation, tone mapping, screenshot export, and `egui` rendering.
- `src/gpu/` & `src/surface/`: `wgpu` device/queue/readback helpers and surface/accumulation targets.

# Build & Test
- Run native app: `cargo run`
- Run unit tests: `cargo test --lib`

# Design choices
- It is important that the app runs and renders fast. That is why everything that can be run on the GPU to make it faster is run on the GPU.
- We want to keep the code as simple as possible. Re-use and generalize as much as possible.