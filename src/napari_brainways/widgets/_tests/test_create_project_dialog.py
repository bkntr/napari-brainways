from dataclasses import replace
from pathlib import Path

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_project_settings import (
    ProjectDocument,
    ProjectSettings,
)
from brainways.utils.image import ImageSizeHW
from brainways.utils.io import ImagePath
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QCheckBox

from napari_brainways.widgets.create_project_dialog import CreateProjectDialog


@fixture
def create_project_dialog(
    qtbot: QtBot, mock_image_path: ImagePath, test_image_size: ImageSizeHW
) -> CreateProjectDialog:
    create_project_dialog = CreateProjectDialog()
    create_project_dialog.add_filename(str(mock_image_path.filename))
    return create_project_dialog


@fixture
def create_project_document(
    qtbot: QtBot, mock_image_path: ImagePath, test_image_size: ImageSizeHW
) -> ProjectDocument:
    return ProjectDocument(
        path=mock_image_path,
        image_size=test_image_size,
        lowres_image_size=test_image_size,
    )


def test_documents(
    create_project_dialog: CreateProjectDialog,
    create_project_document: ProjectDocument,
):
    documents = create_project_dialog.project.documents
    expected = [create_project_document]
    assert documents == expected


def test_ignore(
    create_project_dialog: CreateProjectDialog,
    create_project_document: ProjectDocument,
):
    checkbox: QCheckBox = create_project_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    documents = create_project_dialog.project.documents
    expected = [replace(create_project_document, ignore=True)]
    assert documents == expected


def test_project_path(create_project_dialog: CreateProjectDialog):
    assert create_project_dialog.project_path == Path(".")


def test_edit_project(create_project_document: ProjectDocument, tmpdir):
    project = BrainwaysProject(
        settings=ProjectSettings(atlas="", channel=0),
        documents=[create_project_document],
        project_path=Path(tmpdir),
    )
    dialog = CreateProjectDialog(project=project)
    assert dialog.project == project
    assert dialog.files_table.rowCount() == 1
    assert dialog.project_path == project.project_path


def test_uncheck_check(create_project_dialog: CreateProjectDialog):
    checkbox: QCheckBox = create_project_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    checkbox.setChecked(True)
    assert create_project_dialog.project.documents[0].ignore is False
