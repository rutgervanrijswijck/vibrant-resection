use egui::{ComboBox, Grid, Ui};

use crate::{asset::volume_fraction::VolumeFractionBuffer, controller::icons};

#[derive(Debug, Default)]
pub struct ResectionWidget {
    enabled: bool,
    volume: usize,
}

impl ResectionWidget {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn volume(&self) -> usize {
        self.volume
    }

    pub fn show(&mut self, ui: &mut Ui, volumes: &[VolumeFractionBuffer]) {
        if volumes.is_empty() {
            self.enabled = false;
            ui.vertical_centered(|ui| {
                ui.weak("Open .nii.gz files using the [📂 open] button");
            });
            return;
        }

        if self.volume >= volumes.len() {
            self.volume = volumes.len() - 1;
        }

        Grid::new("ResectionWidgetGrid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Tool").on_hover_text(
                    "Enable the resection tool to click on the selected volume \
                     and add the hit voxel to a resection mask.",
                );
                ui.toggle_value(
                    &mut self.enabled,
                    format!("{} Resect", icons::regular::ERASER),
                )
                .on_hover_text(
                    "Toggle resection mode. While active, left-clicking on the \
                     volume selects the hit voxel instead of rotating the camera.",
                );
                ui.end_row();

                ui.label("Volume")
                    .on_hover_text("Volume on which resection takes place.");
                ComboBox::from_id_salt("ResectionVolume")
                    .selected_text(volumes[self.volume].settings().name.clone())
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (index, volume) in volumes.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.volume,
                                index,
                                volume.settings().name.clone(),
                            );
                        }
                    });
                ui.end_row();
            });
    }
}
