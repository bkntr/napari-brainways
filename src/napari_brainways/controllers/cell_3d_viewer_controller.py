from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import napari.layers
import numpy as np
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas

if TYPE_CHECKING:
    from napari_brainways.widget import BrainwaysUI


class Cell3DViewerController:
    def __init__(self, ui: BrainwaysUI):
        self.ui = ui
        self.input_layer: napari.layers.Image | None = None
        self.points_layer: napari.layers.Points | None = None
        self.atlas_layer: napari.layers.Image | None = None
        self._params: BrainwaysParams | None = None
        self._atlas: BrainwaysAtlas | None = None

    @property
    def name(self) -> str:
        return "3D Cell Viewer"

    def open(self, atlas: BrainwaysAtlas) -> None:
        self._atlas = atlas
        self.atlas_layer = self.viewer.add_image(atlas.reference.numpy(), name="Atlas")
        self.points_layer = self.viewer.add_points(
            size=1, ndim=3, name="Detected Cells"
        )
        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()

    def close(self) -> None:
        self.viewer.layers.remove(self.atlas_layer)
        self.viewer.layers.remove(self.points_layer)
        self.atlas_layer = None
        self.points_layer = None
        self.viewer.dims.ndisplay = 2
        self._atlas = None

    def show_cells(self, cells: np.ndarray) -> None:
        if len(cells) > 0:
            colors = []
            for cell in cells:
                try:
                    struct_id = self._atlas.atlas.structure_from_coords(cell[::-1])
                except IndexError:
                    struct_id = 0
                if struct_id == 0:
                    colors.append([0, 0, 0, 255])
                else:
                    colors.append(
                        self._atlas.atlas.structures[struct_id]["rgb_triplet"] + [255]
                    )
            colors = np.array(colors) / 255

            self.points_layer.data = np.array(cells)[:, ::-1]
            self.points_layer.edge_color = colors
            self.points_layer.selected_data = set()

    @property
    def params(self) -> BrainwaysParams:
        return self._params
