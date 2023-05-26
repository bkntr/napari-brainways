from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Tuple

import napari
import napari.layers
import numpy as np
from brainways.pipeline.brainways_params import BrainwaysParams, CellDetectorParams
from brainways.pipeline.cell_detector import ClaheNormalizer
from napari.qt.threading import FunctionWorker, create_worker

from napari_brainways.controllers.base import Controller
from napari_brainways.utils import update_layer_contrast_limits
from napari_brainways.widgets.cell_detector_widget import CellDetectorWidget

if TYPE_CHECKING:
    from napari_brainways.brainways_ui import BrainwaysUI


class CellDetectorController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.model = None
        self.widget = CellDetectorWidget(self)
        self.widget.hide()

        self.input_layer = None
        self.preview_box_layer: napari.layers.Points | None = None
        self.crop_layer: napari.layers.Image | None = None
        self.cell_mask_layer: napari.layers.Image | None = None
        self._run_lock = False
        self._params = None
        self._worker = None

    @property
    def name(self) -> str:
        return "Cell Detection"

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return params.cell is not None

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return True

    def default_params(
        self, image: np.ndarray, params: BrainwaysParams
    ) -> BrainwaysParams:
        preview_bb = self.selected_bounding_box(
            image, point=(0.5 * image.shape[0], 0.5 * image.shape[1])
        )
        return replace(
            params,
            cell=CellDetectorParams(
                diameter=25.0,
                net_avg=True,
                flow_threshold=0.4,
                mask_threshold=0.0,
                preview_bb=preview_bb,
            ),
        )

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params

        if image is not None:
            self.input_layer.data = image
            update_layer_contrast_limits(self.input_layer)

        params_widget = self.widget.cell_detector_params_widget
        params_widget.diameter.value = params.cell.diameter
        params_widget.net_avg.value = params.cell.net_avg
        params_widget.flow_threshold.value = params.cell.flow_threshold
        params_widget.mask_threshold.value = params.cell.mask_threshold

        x0 = params.cell.preview_bb[0]
        y0 = params.cell.preview_bb[1]
        x1 = params.cell.preview_bb[0] + params.cell.preview_bb[2]
        y1 = params.cell.preview_bb[1] + params.cell.preview_bb[3]
        self.preview_box_layer.data = (
            np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
            * self.input_layer.data.shape
        )
        self.on_click()
        self.set_preview_affine()

        self.ui.viewer.reset_view()

    def load_model(self) -> None:
        from brainways.pipeline.cell_detector import CellDetector

        if self.model is None:
            self.model = CellDetector()

    def open(self) -> None:
        self.widget.show()

        self.input_layer = self.ui.viewer.add_image(
            np.zeros((512, 512), np.uint8),
            name="Input",
        )
        self.input_layer.translate = (0, 0)

        self.preview_box_layer = self.ui.viewer.add_shapes(
            name="Region selector",
            face_color="#ffffff00",
            edge_color="red",
        )
        self.preview_box_layer.mouse_double_click_callbacks.append(self.on_click)

        self.crop_layer = self.ui.viewer.add_image(
            np.zeros((100, 100), np.uint8),
            name="Preview",
        )
        self.cell_mask_layer = self.ui.viewer.add_labels(
            np.zeros((10, 10), np.uint8), name="Cells"
        )
        self.ui.viewer.layers.selection.active = self.preview_box_layer

        self._is_open = True

    def close(self):
        if not self._is_open:
            return

        self.widget.hide()
        self.ui.viewer.layers.remove(self.input_layer)
        self.ui.viewer.layers.remove(self.preview_box_layer)
        self.ui.viewer.layers.remove(self.crop_layer)
        self.ui.viewer.layers.remove(self.cell_mask_layer)

        self._image = None
        self._params = None

        self.input_layer = None
        self.preview_box_layer = None
        self.crop_layer = None
        self.cell_mask_layer = None
        self._image_reader = None
        self._is_open = False

    @property
    def _preview_translate(self):
        self._check_is_open()
        ty = 0
        tx = self.input_layer.data.shape[1]
        return ty, tx

    @property
    def _preview_scale(self):
        self._check_is_open()
        scale = self.input_layer.data.shape[0] / self.crop_layer.data.shape[0]
        return scale, scale

    def set_preview_affine(self):
        self.crop_layer.translate = self._preview_translate
        self.cell_mask_layer.translate = self._preview_translate
        self.crop_layer.scale = self._preview_scale
        self.cell_mask_layer.scale = self._preview_scale

    def _on_cell_detector_returned(self, mask: np.ndarray):
        self.cell_mask_layer.data = mask
        self.cell_mask_layer.visible = True
        self.ui.viewer.layers.selection = {self.preview_box_layer}

    def _on_cell_detector_started(self):
        self._run_lock = True
        self.widget.show_progress_bar()

    def _on_cell_detector_finished(self):
        self._run_lock = False
        self.widget.hide_progress_bar()

    def run_cell_detector_preview_async(
        self,
        diameter: float,
        net_avg: bool,
        flow_threshold: float,
        mask_threshold: float,
    ):
        self._check_is_open()

        self.load_model()
        self._worker: FunctionWorker = create_worker(
            self.model.run_cell_detector,
            self.crop_layer.data,
            normalizer=ClaheNormalizer(),
            # diameter=diameter,
            # net_avg=net_avg,
            # flow_threshold=flow_threshold,
            # mask_threshold=mask_threshold,
        )
        self._worker.returned.connect(self._on_cell_detector_returned)
        self._worker.started.connect(self._on_cell_detector_started)
        self._worker.finished.connect(self._on_cell_detector_finished)
        self._worker.start()

    @property
    def params(self) -> BrainwaysParams:
        params_widget = self.widget.cell_detector_params_widget
        params = replace(
            self._params,
            cell=CellDetectorParams(
                diameter=params_widget.diameter.value,
                net_avg=params_widget.net_avg.value,
                flow_threshold=params_widget.flow_threshold.value,
                mask_threshold=params_widget.mask_threshold.value,
                preview_bb=self.selected_bounding_box(),
            ),
        )
        return params

    def selected_bounding_box(
        self, image: np.ndarray | None = None, point: Tuple[float, float] | None = None
    ):
        """

        :return: x, y, w, h
        """
        # TODO: convert points layer to shapes
        if image is None:
            image = self.input_layer.data

        if point is None:
            box = self.preview_box_layer.data[-1] / image.shape
            x = box[0, 1]
            y = box[0, 0]
            w = box[1, 1] - box[0, 1]
            h = box[2, 0] - box[0, 0]
            return x, y, w, h

        image_height = image.shape[0]
        image_width = image.shape[1]

        y = point[0] / image_height
        x = point[1] / image_width

        w = min(4096 / self.ui.current_document.image_size[1], 1)
        h = min(4096 / self.ui.current_document.image_size[0], 1)

        x0 = min(max(x - w / 2, 0), 1 - w)
        y0 = min(max(y - h / 2, 0), 1 - h)

        return x0, y0, w, h

    def on_click(self, layer=None, event=None):
        if self._run_lock:
            with self.preview_box_layer.events.data.blocker():
                self.preview_box_layer.selected_data = {
                    self.preview_box_layer.data.shape[0] - 1
                }
                self.preview_box_layer.remove_selected()
            return

        if event is not None:
            with self.preview_box_layer.events.data.blocker():
                x, y, w, h = self.selected_bounding_box(point=event.position)
                self.preview_box_layer.data = (
                    np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]])
                    * self.input_layer.data.shape
                )

        x, y, w, h = self.selected_bounding_box()
        x0 = int(round(x * self.ui.current_document.image_size[1]))
        y0 = int(round(y * self.ui.current_document.image_size[0]))
        x1 = int(round((x + w) * self.ui.current_document.image_size[1]))
        y1 = int(round((y + h) * self.ui.current_document.image_size[0]))

        highres_crop = (
            self.ui.current_document.image_reader()
            .get_image_dask_data(
                "YX",
                X=slice(x0, x1),
                Y=slice(y0, y1),
                C=self.ui.current_subject.settings.channel,
            )
            .compute()
        )
        self.crop_layer.data = highres_crop
        update_layer_contrast_limits(
            self.crop_layer, contrast_limits_quantiles=(0, 0.98)
        )
        self.cell_mask_layer.data = np.zeros_like(self.crop_layer.data, dtype=np.uint8)
        self.set_preview_affine()
