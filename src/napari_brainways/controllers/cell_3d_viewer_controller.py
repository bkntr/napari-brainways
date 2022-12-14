from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import napari.layers
import numpy as np
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.utils.atlas.brainways_atlas import BrainwaysAtlas
from brainways.utils.cells import get_cell_struct_ids, get_struct_colors

from napari_brainways.controllers.base import Controller
from napari_brainways.utils import update_layer_contrast_limits

if TYPE_CHECKING:
    from napari_brainways.brainways_ui import BrainwaysUI


class Cell3DViewerController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.input_layer: napari.layers.Image | None = None
        self.points_layer: napari.layers.Points | None = None
        self.atlas_layer: napari.layers.Image | None = None
        self._params: BrainwaysParams | None = None
        self._atlas: BrainwaysAtlas | None = None
        # TODO: ability to switch between modes
        self._3d_view_mode = True

    def open(self) -> None:
        self._atlas = self.ui.project.atlas
        self.input_layer = self.ui.viewer.add_image(np.zeros((10, 10)), name="Image")
        self.atlas_layer = self.ui.viewer.add_image(
            self._atlas.reference.numpy(),
            name="Atlas",
            rendering="attenuated_mip",
            attenuation=0.5,
        )
        self.points_layer = self.ui.viewer.add_points(
            size=1, ndim=3, name="Detected Cells"
        )
        self.ui.viewer.dims.ndisplay = 3
        self.ui.viewer.reset_view()

    def close(self) -> None:
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.atlas_layer)
        self.ui.viewer.layers.remove(self.points_layer)
        self.atlas_layer = None
        self.points_layer = None
        self.ui.viewer.dims.ndisplay = 2
        self._atlas = None

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params
        if self._3d_view_mode:
            self.show_3d()
        else:
            self.show_2d(image=image, from_ui=from_ui)

    def show_3d(self):
        self.input_layer.visible = False
        self.atlas_layer.visible = True
        self.ui.viewer.dims.ndisplay = 3

        self.ui.viewer.layers.remove(self.points_layer)
        self.points_layer = self.ui.viewer.add_points(
            size=1, ndim=3, name="Detected Cells"
        )

        subject = self.ui.current_subject
        all_cells = subject.get_cells_on_atlas()
        if all_cells is not None:
            # TODO: get struct ids from annotation, not from brainglobe
            struct_ids = get_cell_struct_ids(all_cells, self._atlas.brainglobe_atlas)
            colors = get_struct_colors(struct_ids, self._atlas.brainglobe_atlas)
            self.points_layer.data = all_cells[["z", "y", "x"]].values
            self.points_layer.face_color = colors
            self.points_layer.edge_color = colors
            self.points_layer.selected_data = set()
        else:
            self.points_layer.data = []

    def show_2d(self, image: np.ndarray | None = None, from_ui: bool = False):
        if image is None:
            return

        self.input_layer.visible = True
        self.atlas_layer.visible = False
        self.ui.viewer.dims.ndisplay = 2

        self.ui.viewer.layers.remove(self.points_layer)
        self.points_layer = self.ui.viewer.add_points(
            size=max(image.shape) * 0.002, ndim=2, name="Detected Cells"
        )

        subject = self.ui.current_subject
        document = self.ui.current_document

        # TODO: read image from disk with options for highres and other channels
        self.input_layer.data = image
        update_layer_contrast_limits(self.input_layer)

        cells_atlas = subject.get_cells_on_atlas([document])
        if cells_atlas is not None:
            cells = subject.get_valid_cells(document)
            # TODO: get struct ids from annotation, not from brainglobe
            struct_ids = get_cell_struct_ids(
                cells_atlas, subject.atlas.brainglobe_atlas
            )
            colors = get_struct_colors(struct_ids, subject.atlas.brainglobe_atlas)
            cell_xy = cells[["y", "x"]].values * [image.shape[0], image.shape[1]]
            self.points_layer.data = cell_xy
            self.points_layer.face_color = colors
            self.points_layer.edge_color = colors
            self.points_layer.selected_data = set()
        else:
            self.points_layer.data = []

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        return params

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return True

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    @property
    def params(self) -> BrainwaysParams:
        return self._params

    @property
    def name(self) -> str:
        return "3D Cell Viewer"
