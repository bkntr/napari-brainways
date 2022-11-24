import os
import pickle
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest
import torch
from bg_atlasapi.structure_class import StructuresDict
from brainways.pipeline.brainways_params import (
    AffineTransform2DParams,
    AtlasRegistrationParams,
    BrainwaysParams,
    TPSTransformParams,
)
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ProjectSettings, SliceInfo
from brainways.utils.atlas.brainways_atlas import AtlasSlice, BrainwaysAtlas
from brainways.utils.image import ImageSizeHW
from brainways.utils.io_utils import ImagePath
from brainways.utils.io_utils.readers.base import ImageReader
from PIL import Image
from pytest import fixture
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QApplication

from napari_brainways.brainways_ui import BrainwaysUI
from napari_brainways.test_utils import worker_join


@fixture(scope="session", autouse=True)
def env_config():
    """
    Configure environment variables needed for the test session
    """

    # This makes QT render everything offscreen and thus prevents
    # any Modals / Dialogs or other Widgets being rendered on the screen while running unit tests
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    yield

    os.environ.pop("QT_QPA_PLATFORM")


@fixture(autouse=True)
def setup_qt(qapp: QApplication):
    # the pytestqt.qapp fixture sets up the QApplication required to run QT code
    # see https://pytest-qt.readthedocs.io/en/latest/reference.html
    yield


@fixture
def napari_viewer(make_napari_viewer):
    return make_napari_viewer()


@fixture
def app(
    napari_viewer,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_atlas: BrainwaysAtlas,
    monkeypatch,
) -> BrainwaysUI:
    monkeypatch.setattr(BrainwaysAtlas, "load", Mock(return_value=mock_atlas))
    app = BrainwaysUI(napari_viewer)
    yield app


@fixture
def opened_app(
    qtbot: QtBot,
    app: BrainwaysUI,
    test_data: Tuple[np.ndarray, AtlasSlice],
    project_path: Path,
):
    worker = app.open_project_async(project_path)
    worker_join(worker, qtbot)
    return app


@fixture(autouse=True)
def seed():
    np.random.seed(0)


@fixture
def mock_atlas(test_data: Tuple[np.ndarray, AtlasSlice]) -> BrainwaysAtlas:
    test_image, test_atlas_slice = test_data
    ATLAS_SIZE = 32
    ATLAS_DEPTH = 10
    mock_atlas = create_autospec(BrainwaysAtlas)
    mock_atlas.bounding_boxes = [(0, 0, ATLAS_SIZE, ATLAS_SIZE)] * ATLAS_DEPTH
    mock_atlas.shape = (ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.reference = torch.rand(ATLAS_DEPTH, ATLAS_SIZE, ATLAS_SIZE)
    mock_atlas.atlas = Mock()
    mock_atlas.atlas.structure_from_coords = Mock(return_value=10)
    mock_atlas.atlas.resolution = (1, 2, 3)
    mock_atlas.atlas.atlas_name = "MOCK_ATLAS"
    structures_list = [
        {
            "name": "root",
            "acronym": "root",
            "id": 1,
            "structure_id_path": [1],
            "rgb_triplet": [0, 0, 0],
            "mesh_filename": Path("/"),
        },
        {
            "name": "test_region",
            "acronym": "TEST",
            "id": 10,
            "structure_id_path": [1, 10],
            "rgb_triplet": [255, 255, 255],
            "mesh_filename": Path("/"),
        },
    ]
    structures = StructuresDict(structures_list=structures_list)
    mock_atlas.atlas.structures = structures
    mock_atlas.slice = Mock(return_value=test_atlas_slice)
    return mock_atlas


@fixture(scope="session")
def test_data() -> Tuple[np.ndarray, AtlasSlice]:
    npz = np.load(str(Path(__file__).parent.parent.parent / "data/test_data.npz"))
    input = npz["input"]
    reference = npz["atlas_slice_reference"]
    annotation = npz["atlas_slice_annotation"]
    hemispheres = npz["atlas_slice_hemispheres"]
    atlas_slice = AtlasSlice(
        reference=torch.as_tensor(reference),
        annotation=torch.as_tensor(annotation),
        hemispheres=torch.as_tensor(hemispheres),
    )
    return input, atlas_slice


@fixture(scope="session")
def test_image_size(test_data: Tuple[np.ndarray, AtlasSlice]) -> ImageSizeHW:
    input, atlas_size = test_data
    return input.shape


@fixture(autouse=True, scope="session")
def image_reader_mock(test_data: Tuple[np.ndarray, AtlasSlice]):
    mock_image_reader = create_autospec(ImageReader)
    test_image, test_atlas_slice = test_data
    HEIGHT = test_image.shape[0]
    WIDTH = test_image.shape[1]
    mock_image_reader.read_image.return_value = test_image
    mock_image_reader.scene_bb = (0, 0, WIDTH, HEIGHT)

    mock_get_scenes = Mock(return_value=[0])

    with patch(
        "brainways.utils.io_utils.readers.get_reader", return_value=mock_image_reader
    ), patch(
        "brainways.utils.io_utils.readers.get_scenes", return_value=mock_get_scenes
    ):
        yield


@pytest.fixture
def mock_image_path(test_data: Tuple[np.ndarray, AtlasSlice], tmpdir) -> ImagePath:
    image, _ = test_data
    image_path = ImagePath(str(tmpdir / "image.jpg"), scene=0)
    Image.fromarray(image).save(image_path.filename)
    return image_path


@pytest.fixture
def mock_subject_documents(
    mock_image_path: ImagePath, test_data: Tuple[np.ndarray, AtlasSlice]
) -> List[SliceInfo]:
    test_image, test_atlas_slice = test_data
    image_height = test_image.shape[0]
    image_width = test_image.shape[1]
    tps_points = (np.random.rand(10, 2) * (image_width, image_height)).astype(
        np.float32
    )

    params = BrainwaysParams(
        atlas=AtlasRegistrationParams(ap=5),
        affine=AffineTransform2DParams(),
        tps=TPSTransformParams(
            points_src=tps_points,
            points_dst=tps_points,
        ),
    )
    documents = []
    for i in range(3):
        doc_image_filename_name = f"{Path(mock_image_path.filename).stem}_{i}.jpg"
        doc_image_filename = Path(mock_image_path.filename).with_name(
            doc_image_filename_name
        )
        shutil.copy(mock_image_path.filename, doc_image_filename)
        doc_image_path = replace(mock_image_path, filename=str(doc_image_filename))
        documents.append(
            SliceInfo(
                path=doc_image_path,
                image_size=(image_height, image_width),
                lowres_image_size=(image_height, image_width),
                params=params,
                ignore=i == 0,
            )
        )
    return documents


@pytest.fixture
def mock_project_settings() -> ProjectSettings:
    return ProjectSettings(atlas="MOCK_ATLAS", channel=0)


def _create_subject(
    subject_dir: Path, project_settings: ProjectSettings, slice_infos: List[SliceInfo]
) -> Path:
    subject_dir.mkdir()
    serialized_subject_settings = asdict(project_settings)
    serialized_subject_documents = [asdict(doc) for doc in slice_infos]
    with open(subject_dir / "brainways.bin", "wb") as f:
        pickle.dump((serialized_subject_settings, serialized_subject_documents), f)
    return subject_path


@pytest.fixture
def subject_path(
    tmpdir,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
) -> Path:
    subject_path = Path(tmpdir) / "project/subject1/brainways.bin"
    subject_path.parent.mkdir(parents=True)
    serialized_subject_settings = asdict(mock_project_settings)
    serialized_subject_documents = [asdict(doc) for doc in mock_subject_documents]
    with open(subject_path, "wb") as f:
        pickle.dump((serialized_subject_settings, serialized_subject_documents), f)
    yield subject_path


@pytest.fixture
def project_path(
    tmpdir,
    mock_project_settings: ProjectSettings,
    mock_subject_documents: List[SliceInfo],
) -> Path:
    project_dir = Path(tmpdir) / "project"
    project_path = project_dir / "project.bwp"
    project_dir.mkdir()
    _create_subject(
        project_dir / "subject1",
        project_settings=mock_project_settings,
        slice_infos=mock_subject_documents,
    )
    _create_subject(
        project_dir / "subject2",
        project_settings=mock_project_settings,
        slice_infos=mock_subject_documents,
    )
    serialized_project_settings = asdict(mock_project_settings)
    with open(project_path, "wb") as f:
        pickle.dump(serialized_project_settings, f)
    yield project_path


@pytest.fixture
def brainways_subject(
    subject_path: Path,
    test_data: Tuple[np.ndarray, AtlasSlice],
    mock_atlas: BrainwaysAtlas,
) -> BrainwaysSubject:
    brainways_subject = BrainwaysSubject.open(subject_path)
    brainways_subject.atlas = mock_atlas
    for document in brainways_subject.documents:
        brainways_subject.read_lowres_image(document)
    return brainways_subject
