from dataclasses import replace

from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import SliceInfo
from brainways.utils.image import ImageSizeHW, get_resize_size
from brainways.utils.io_utils import ImagePath
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QCheckBox

from napari_brainways.test_utils import worker_join
from napari_brainways.widgets.create_subject_dialog import CreateSubjectDialog


@fixture
def create_subject_dialog(
    qtbot: QtBot,
    mock_project: BrainwaysProject,
    mock_image_path: ImagePath,
    test_image_size: ImageSizeHW,
) -> CreateSubjectDialog:
    create_subject_dialog = CreateSubjectDialog(mock_project)
    create_subject_dialog.new_subject("test_subject")
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
    create_subject_dialog: CreateSubjectDialog,
    create_subject_document: SliceInfo,
):
    documents = create_subject_dialog.subject.documents
    expected = [create_subject_document]
    assert documents == expected


def test_ignore(
    create_subject_dialog: CreateSubjectDialog,
    create_subject_document: SliceInfo,
):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    documents = create_subject_dialog.subject.documents
    expected = [replace(create_subject_document, ignore=True)]
    assert documents == expected


def test_edit_subject(qtbot: QtBot, mock_project: BrainwaysProject, tmpdir):
    dialog = CreateSubjectDialog(project=mock_project)
    worker = dialog.edit_subject_async(subject_index=1, document_index=1)
    worker_join(worker, qtbot)
    assert dialog.subject == mock_project.subjects[1]
    assert dialog.files_table.rowCount() == len(mock_project.subjects[1].documents)
    selected_row = dialog.files_table.selectionModel().selectedRows()[0].row()
    assert selected_row == 1


def test_uncheck_check(create_subject_dialog: CreateSubjectDialog):
    checkbox: QCheckBox = create_subject_dialog.files_table.cellWidget(0, 0)
    checkbox.setChecked(False)
    checkbox.setChecked(True)
    assert create_subject_dialog.subject.documents[0].ignore is False
