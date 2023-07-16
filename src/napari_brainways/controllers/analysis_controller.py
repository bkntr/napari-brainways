from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import napari
import napari.layers
import numpy as np
import pandas as pd
from brainways.pipeline.brainways_params import BrainwaysParams
from napari.qt.threading import FunctionWorker

from napari_brainways.controllers.base import Controller
from napari_brainways.utils import update_layer_contrast_limits
from napari_brainways.widgets.analysis_widget import AnalysisWidget

if TYPE_CHECKING:
    from napari_brainways.brainways_ui import BrainwaysUI


class AnalysisController(Controller):
    def __init__(self, ui: BrainwaysUI):
        super().__init__(ui)
        self.atlas_layer: napari.layers.Image | None = None
        self.annotations_layer: napari.layers.Image | None = None
        self._params: BrainwaysParams | None = None
        self.widget = AnalysisWidget(self)

    @property
    def name(self) -> str:
        return "Analysis"

    def default_params(self, image: np.ndarray, params: BrainwaysParams):
        return params

    def run_model(self, image: np.ndarray, params: BrainwaysParams) -> BrainwaysParams:
        return params

    @staticmethod
    def has_current_step_params(params: BrainwaysParams) -> bool:
        return True

    @staticmethod
    def enabled(params: BrainwaysParams) -> bool:
        return True

    def open(self) -> None:
        if self._is_open:
            return

        self.atlas_layer = self.ui.viewer.add_image(
            self.ui.project.atlas.reference.numpy(),
            name="Atlas",
            rendering="attenuated_mip",
            attenuation=0.5,
        )
        self.annotations_layer = self.ui.viewer.add_labels(
            self.ui.project.atlas.annotation.numpy().astype(np.int32),
            name="Structures",
        )
        self.contrast_layer = self.ui.viewer.add_image(
            np.zeros_like(self.ui.project.atlas.annotation),
            name="Contrast",
            rendering="attenuated_mip",
            attenuation=0.5,
            colormap="inferno",
            blending="additive",
        )
        self.annotations_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self._is_open = True

    def on_mouse_move(self, _layer, event):
        struct_id = self.annotations_layer.get_value(event.position, world=True)
        if struct_id and struct_id in self.pipeline.atlas.brainglobe_atlas.structures:
            struct_name = self.pipeline.atlas.brainglobe_atlas.structures[struct_id][
                "name"
            ]
        else:
            struct_name = ""
        _layer.help = struct_name

    def close(self) -> None:
        self.ui.viewer.layers.remove(self.atlas_layer)
        self.ui.viewer.layers.remove(self.annotations_layer)
        self.ui.viewer.layers.remove(self.contrast_layer)
        self.atlas_layer = None
        self.annotations_layer = None
        self.contrast_layer = None
        self._params = None
        self._is_open = False

    def show(
        self,
        params: BrainwaysParams,
        image: np.ndarray | None = None,
        from_ui: bool = False,
    ) -> None:
        self._params = params

    def plot_anova(self, anova_df: pd.DataFrame):
        atlas = self.ui.project.atlas
        annotation = self.ui.project.atlas.annotation.numpy()
        annotation_anova = np.zeros_like(annotation)
        for structure, row in anova_df[anova_df["reject"]].iterrows():
            struct_id = atlas.brainglobe_atlas.structures[structure]["id"]
            struct_mask = annotation == struct_id
            annotation_anova[struct_mask] = row["F"]
        self.contrast_layer.data = annotation_anova
        update_layer_contrast_limits(self.contrast_layer)

        self.contrast_layer.visible = True
        self.annotations_layer.visible = False

    def plot_posthoc(self, posthoc_df: pd.DataFrame, contrast: str, pvalue: float):
        atlas = self.ui.project.atlas
        annotation = self.ui.project.atlas.annotation.numpy()
        annotation_anova = np.zeros_like(annotation)
        for structure, row in posthoc_df[posthoc_df[contrast] <= pvalue].iterrows():
            struct_id = atlas.brainglobe_atlas.structures[structure]["id"]
            struct_mask = annotation == struct_id
            annotation_anova[struct_mask] = -np.log(row[contrast])
        self.contrast_layer.data = annotation_anova
        update_layer_contrast_limits(self.contrast_layer)

        self.contrast_layer.visible = True
        self.annotations_layer.visible = False

    def run_calculate_results_async(
        self,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
    ) -> FunctionWorker:
        return self.ui.do_work_async(
            self.ui.project.calculate_results_iter,
            min_region_area_um2=min_region_area_um2,
            cells_per_area_um2=cells_per_area_um2,
            min_cell_size_um=min_cell_size_um,
            max_cell_size_um=max_cell_size_um,
            progress_label="Calculating Brainways Results...",
            progress_max_value=len(self.ui.project.subjects),
        )

    def run_contrast_analysis(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        pvalue: float,
        multiple_comparisons_method: str,
    ):
        anova_df, posthoc_df = self.ui.project.calculate_contrast(
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
        )

        # self.plot_anova(anova_df)
        self.plot_posthoc(posthoc_df, contrast="outgroup-ingroup", pvalue=0.05)

    @property
    def params(self) -> BrainwaysParams:
        return self._params
