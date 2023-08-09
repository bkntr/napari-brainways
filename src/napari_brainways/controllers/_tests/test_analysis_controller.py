from typing import Tuple
from unittest.mock import Mock

import pandas as pd
from pytest import fixture
from pytestqt.qtbot import QtBot

from napari_brainways.brainways_ui import BrainwaysUI
from napari_brainways.controllers.analysis_controller import AnalysisController
from napari_brainways.test_utils import worker_join


@fixture
def app_on_analysis(
    qtbot: QtBot, opened_app: BrainwaysUI
) -> Tuple[BrainwaysUI, AnalysisController]:
    tps_step_index = [
        isinstance(step, AnalysisController) for step in opened_app.steps
    ].index(True)
    opened_app.set_step_index_async(tps_step_index, run_async=False)
    controller: AnalysisController = opened_app.current_step
    return opened_app, controller


def test_analysis_controller_run_pls_analysis(
    qtbot: QtBot, app_on_analysis: Tuple[BrainwaysUI, AnalysisController]
):
    app, controller = app_on_analysis
    for subject in app.project.subjects:
        n = 6
        subject.cell_count_summary = Mock(
            return_value=pd.DataFrame(
                {
                    "condition1": [subject.subject_info.conditions["condition1"]] * n,
                    "animal_id": [subject.subject_info.name] * n,
                    "acronym": [str(i) for i in range(n)],
                    "name": ["a"] * n,
                    "is_parent_structure": [False] * n,
                    "is_gray_matter": [True] * n,
                    "total_area_um2": [10.0] * n,
                    "cells": [10.0] * n,
                }
            )
        )

    worker_join(controller.run_calculate_results_async(), qtbot)
    controller.run_pls_analysis(
        condition_col="condition1", values_col="cells", min_group_size=1, alpha=1.0
    )
