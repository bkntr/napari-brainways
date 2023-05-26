import shutil
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo
from brainways.utils.io_utils import ImagePath
from pytest import fixture
from pytestqt.qtbot import QtBot

from napari_brainways.brainways_ui import BrainwaysUI
from napari_brainways.controllers.base import Controller
from napari_brainways.test_utils import randomly_modified_params, worker_join


@fixture(params=[0, 1])
def image_index(request):
    return request.param


@fixture(params=[0, 1])
def subject_index(request):
    return request.param


STEP_INDICES = [0, 1, 2, 3, 4, 5]


@fixture(params=STEP_INDICES)
def step_index(opened_app: BrainwaysUI, request) -> int:
    assert len(STEP_INDICES) == len(opened_app.steps)
    return request.param


@fixture
def step(opened_app: BrainwaysUI, step_index: int):
    return opened_app.steps[step_index]


@fixture
def subject_doc() -> SliceInfo:
    return SliceInfo(
        path=ImagePath("/"),
        image_size=(3840, 5120),
        lowres_image_size=(384, 512),
        params=BrainwaysParams(),
    )


def test_app_init(app: BrainwaysUI):
    pass


def test_steps_are_loading(qtbot: QtBot, opened_app: BrainwaysUI, step: Controller):
    opened_app.set_step_index_async(opened_app.steps.index(step), run_async=False)


def test_next_image_prev_image_keeps_changed_params(
    qtbot: QtBot, opened_app: BrainwaysUI, step: Controller
):
    # set step
    opened_app.set_step_index_async(opened_app.steps.index(step), run_async=False)

    # modify params
    current_params = opened_app.current_params
    first_modification = randomly_modified_params(current_params)
    assert current_params != first_modification
    step.show(first_modification)

    # go next image
    worker = opened_app.next_image()
    worker_join(worker, qtbot)

    # modify params again
    second_modification = randomly_modified_params(opened_app.current_params)
    step.show(second_modification)

    # go prev image
    worker = opened_app.prev_image()
    worker_join(worker, qtbot)

    # assert that params of first image didn't change
    opened_app.persist_current_params()
    assert opened_app.current_params == first_modification


@pytest.mark.skip
def test_run_workflow(qtbot: QtBot, opened_app: BrainwaysUI):
    worker = opened_app.run_workflow_async()
    worker_join(worker, qtbot)


def test_open_project(
    qtbot: QtBot,
    app: BrainwaysUI,
    project_path: Path,
):
    assert app.project is None
    worker = app.open_project_async(project_path)
    worker_join(worker, qtbot)
    assert isinstance(app.project, BrainwaysProject)
    assert isinstance(app.current_subject, BrainwaysSubject)


def test_open_project_without_subjects(
    qtbot: QtBot, app: BrainwaysUI, project_path: Path
):
    for subject_dir in project_path.parent.glob("subject*"):
        shutil.rmtree(subject_dir)
    assert app.project is None
    worker = app.open_project_async(project_path)
    worker_join(worker, qtbot)
    assert isinstance(app.project, BrainwaysProject)
    assert app._current_valid_subject_index is None


def test_set_subject_index_async(
    qtbot: QtBot,
    opened_app: BrainwaysUI,
    subject_index: int,
):
    worker = opened_app.set_subject_index_async(subject_index)
    worker_join(worker, qtbot)
    assert opened_app.current_subject == opened_app.project.subjects[subject_index]


@pytest.mark.skip
def test_save_load_subject(
    qtbot: QtBot,
    opened_app: BrainwaysUI,
    step_index: int,
    image_index: int,
    tmpdir,
):
    opened_app.set_step_index_async(step_index, run_async=False)
    worker = opened_app.set_document_index_async(image_index)
    worker_join(worker, qtbot)
    save_path = Path(tmpdir) / "test"
    docs = opened_app.current_subject.documents
    opened_app.save_subject()
    opened_app.current_subject.documents = []
    worker = opened_app.open_subject_async(save_path)
    worker_join(worker, qtbot)
    assert opened_app.current_subject.documents == docs


@pytest.mark.skip
def test_save_after_run_workflow(
    qtbot: QtBot,
    opened_app: BrainwaysUI,
    tmpdir,
):
    worker = opened_app.run_workflow_async()
    worker_join(worker, qtbot)
    save_path = Path(tmpdir) / "test"
    docs = opened_app.current_subject.documents
    opened_app.save_subject()
    opened_app.all_documents = []
    worker = opened_app.open_subject_async(save_path)
    worker_join(worker, qtbot)
    assert opened_app.current_subject.documents == docs


@fixture
def app_batch_run_model(
    qtbot: QtBot,
    opened_app: BrainwaysUI,
    step: Controller,
    step_index: int,
    image_index: int,
) -> Tuple[BrainwaysUI, BrainwaysParams]:
    opened_app.set_step_index_async(step_index, run_async=False)
    worker = opened_app.set_document_index_async(image_index)
    worker_join(worker, qtbot)
    modified_params = randomly_modified_params(opened_app.current_params)
    step.run_model = Mock(return_value=modified_params)
    worker = opened_app.batch_run_model_async()
    worker_join(worker, qtbot)
    return opened_app, modified_params


@pytest.mark.skip
def test_batch_run_model_works(
    app_batch_run_model: Tuple[BrainwaysUI, BrainwaysParams], step: Controller
):
    app, modified_params = app_batch_run_model
    for doc in app.documents:
        assert doc.params == modified_params


@pytest.mark.skip
def test_batch_run_model_ends_with_last_image(
    app_batch_run_model: Tuple[BrainwaysUI, BrainwaysParams]
):
    app, modified_params = app_batch_run_model
    assert app._current_valid_document_index == len(app.documents) - 1


@pytest.mark.skip
def test_export_cells_to_csv(opened_app: BrainwaysUI, tmpdir):
    cells = np.array([[0, 0, 0], [1, 1, 0]])
    opened_app.all_documents = [
        SliceInfo(
            path=ImagePath("/a"),
            image_size=(10, 10),
            lowres_image_size=(10, 10),
            params=BrainwaysParams(),
            region_areas={0: 1},
            cells=cells,
        ),
        SliceInfo(
            path=ImagePath("/b"),
            image_size=(10, 10),
            lowres_image_size=(10, 10),
            params=BrainwaysParams(),
            region_areas={0: 1},
            cells=cells,
        ),
    ]

    cells_path = Path(tmpdir) / "cells.csv"
    opened_app.export_cells(cells_path)
    df = pd.read_csv(cells_path)
    assert df.shape == (2, 2)


def test_autosave_on_set_image_index(qtbot: QtBot, opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    worker_join(opened_app.set_document_index_async(image_index=1), qtbot)
    opened_app.save_subject.assert_called_once()


def test_autosave_on_set_step_index(qtbot: QtBot, opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    opened_app.set_step_index_async(step_index=1, run_async=False)
    opened_app.save_subject.assert_called_once()


@pytest.mark.skip
def test_autosave_on_close(qtbot: QtBot, opened_app: BrainwaysUI):
    opened_app.save_subject = Mock()
    opened_app.viewer.close()
    opened_app.save_subject.assert_called_once()


@pytest.mark.skip
def test_import_cells(qtbot: QtBot, opened_app: BrainwaysUI, tmpdir):
    # for document in opened_app.documents:
    #     assert document.cells is None

    cells = np.random.rand(len(opened_app.documents), 3, 2)

    # create cells csvs
    root = Path(tmpdir)
    for i, document in enumerate(opened_app.documents):
        csv_filename = (
            f"{Path(document.path.filename).stem}_scene{document.path.scene}.csv"
        )
        df = pd.DataFrame({"centroid-0": cells[i, :, 0], "centroid-1": cells[i, :, 1]})
        df.to_csv(root / csv_filename)
    worker = opened_app.import_cells_async(root)
    worker_join(worker, qtbot)

    for i, document in enumerate(opened_app.documents):
        assert np.allclose(document.cells, cells[i])
