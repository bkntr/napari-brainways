from dataclasses import replace
from pathlib import Path

from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.image import ImageSizeHW, get_resize_size
from brainways.utils.io_utils import ImagePath
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QCheckBox

from napari_brainways.test_utils import worker_join
from napari_brainways.widgets.create_subject_dialog import CreateProjectDialog


@fixture
def create_subject_dialog(
    qtbot: QtBot, mock_image_path: ImagePath, test_image_size: ImageSizeHW
) -> CreateProjectDialog:
    create_subject_dialog = CreateProjectDialog()
    worker = create_subject_dialog.add_filenames_async([str(mock_image_path.filename)])
    worker_join(worker, qtbot)
    worker_join(create_subject_dialog._add_documents_worker, qtbot)
    return create_subject_dialog


@fixture
def create_subject_document(
    qtbot: QtBot, mock_image_path: ImagePath, test_image_size: ImageSizeHW
) -> SliceInfo:
    return SliceInfo(
        path=mock_image_path,
        image_size=test_image_size,
        lowres_image_size=get_resize_size(
            test_image_size, (1024, 1024), keep_aspect=True
        ),
    )


def test_documents(
    create_subject_dialog: CreateProjectDialog,
    create_subject_document: SliceInfo,
):
    documents = create_subject_dialog.subject.documents
    expected = [create_subject_document]
    assert documents == expected


def test_ignore(
    create_subject_dialog: CreateProjectDialog,
    create_subject_document: SliceInfo,
):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    documents = create_subject_dialog.subject.documents
    expected = [replace(create_subject_document, ignore=True)]
    assert documents == expected


def test_subject_path(create_subject_dialog: CreateProjectDialog):
    assert (
        create_subject_dialog.subject_path == create_subject_dialog.subject.subject_path
    )


def test_edit_subject(qtbot: QtBot, create_subject_document: SliceInfo, tmpdir):
    subject = BrainwaysSubject(
        settings=ProjectSettings(atlas="", channel=0),
        documents=[create_subject_document],
        subject_path=Path(tmpdir / "new"),
    )
    dialog = CreateProjectDialog(subject=subject)
    worker_join(dialog._add_documents_worker, qtbot)
    assert dialog.subject == subject
    assert dialog.files_table.rowCount() == 1
    assert dialog.subject_path == subject.subject_path


def test_uncheck_check(create_subject_dialog: CreateProjectDialog):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    checkbox.setChecked(True)
    assert create_subject_dialog.subject.documents[0].ignore is False
