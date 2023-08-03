from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

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
        self._condition: str | None = None
        self._anova_df: pd.DataFrame | None = None
        self._posthoc_df: pd.DataFrame | None = None
        self._show_mode: str | None = None
        self._contrast: str | None = None
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
        self._annotations = self.ui.project.atlas.annotation.numpy().astype(np.int32)
        self.annotations_layer = self.ui.viewer.add_labels(
            self._annotations,
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
        self.points_layer = self.ui.viewer.add_points(
            data=np.array([[0.0, 0.0]]),
            features={"string": np.array(["-"])},
            name="Legend",
            size=1,
            text={
                "string": "{string}",
                "anchor": "UPPER_LEFT",
                "color": "cyan",
            },
        )

        self.atlas_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self.annotations_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self.contrast_layer.mouse_move_callbacks.append(self.on_mouse_move)
        self.points_layer.mouse_move_callbacks.append(self.on_mouse_move)

        self._is_open = True

    def on_mouse_move(self, _layer, event):
        _ = self.annotations_layer.extent
        data_position = self.annotations_layer.world_to_data(event.position)
        data_position = tuple(int(round(c)) for c in data_position)
        if all(0 <= c < s for c, s in zip(data_position, self._annotations.shape)):
            struct_id = self._annotations[data_position]
        else:
            struct_id = 0

        string = ""

        if struct_id and struct_id in self.pipeline.atlas.brainglobe_atlas.structures:
            struct_name = self.pipeline.atlas.brainglobe_atlas.structures[struct_id][
                "name"
            ]
            string = struct_name

        if self.current_show_mode:
            minus_log_pvalue = self.contrast_layer.get_value(event.position, world=True)
            if minus_log_pvalue:
                pvalue = np.exp(-minus_log_pvalue).round(5)
                string += f" (p={pvalue:.5})"

        self.points_layer.features = {"string": np.array([string])}
        self.points_layer.data = np.array([event.position[1:]])

    def close(self) -> None:
        self.ui.viewer.layers.remove(self.atlas_layer)
        self.ui.viewer.layers.remove(self.annotations_layer)
        self.ui.viewer.layers.remove(self.contrast_layer)
        self.ui.viewer.layers.remove(self.points_layer)
        self.atlas_layer = None
        self.annotations_layer = None
        self.contrast_layer = None
        self.points_layer = None
        self._params = None
        self._condition = None
        self._anova_df = None
        self._posthoc_df = None
        self._show_mode = None
        self._contrast = None
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
        self._anova_df, self._posthoc_df = self.ui.project.calculate_contrast(
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            pvalue=pvalue,
            multiple_comparisons_method=multiple_comparisons_method,
        )
        self._condition = condition_col
        self.show_contrast("anova")

    def run_pls_analysis(
        self,
        condition_col: str,
        values_col: str,
        min_group_size: int,
        alpha: float,
    ):
        self.ui.project.calculate_pls_analysis(
            condition_col=condition_col,
            values_col=values_col,
            min_group_size=min_group_size,
            alpha=alpha,
        )

    def show_contrast(
        self,
        mode: Literal["anova", "posthoc"],
        contrast: str | None = None,
        pvalue: float | None = None,
    ):
        if mode == "anova":
            self.plot_anova(self._anova_df)
        elif mode == "posthoc":
            assert contrast is not None
            assert pvalue is not None
            self.plot_posthoc(self._posthoc_df, contrast=contrast, pvalue=pvalue)
            self._contrast = contrast
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._show_mode = mode

    @property
    def current_condition(self) -> str | None:
        return self._condition

    @property
    def current_show_mode(self) -> str | None:
        return self._show_mode

    @property
    def current_contrast(self) -> str | None:
        return self._contrast

    @property
    def possible_contrasts(self) -> List[str]:
        result = self.ui.project.possible_contrasts(self._condition)
        return ["-".join(c) for c in result]

    @property
    def params(self) -> BrainwaysParams:
        return self._params
