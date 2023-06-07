from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import napari
import numpy as np
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import ExcelMode, SliceInfo
from brainways.utils.cell_detection_importer.cell_detection_importer import (
    CellDetectionImporter,
)
from napari.qt.threading import FunctionWorker, GeneratorWorker, create_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari_brainways.controllers.affine_2d_controller import Affine2DController
from napari_brainways.controllers.annotation_viewer_controller import (
    AnnotationViewerController,
)
from napari_brainways.controllers.cell_3d_viewer_controller import (
    Cell3DViewerController,
)
from napari_brainways.controllers.cell_detector_controller import CellDetectorController
from napari_brainways.controllers.registration_controller import RegistrationController
from napari_brainways.controllers.tps_controller import TpsController
from napari_brainways.widgets.workflow_widget import WorkflowView


class BrainwaysUI(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        self.viewer = napari_viewer

        self.registration_controller = RegistrationController(self)
        self.affine_2d_controller = Affine2DController(self)
        self.tps_controller = TpsController(self)
        self.annotation_viewer_controller = AnnotationViewerController(self)
        self.cell_detector_controller = CellDetectorController(self)
        self.cell_viewer_controller = Cell3DViewerController(self)

        self.steps = [
            self.registration_controller,
            self.affine_2d_controller,
            self.tps_controller,
            self.annotation_viewer_controller,
            self.cell_detector_controller,
            self.cell_viewer_controller,
        ]

        self.project: Optional[BrainwaysProject] = None
        self._current_valid_subject_index: Optional[int] = None
        self._current_valid_document_index: Optional[int] = None
        self._current_step_index: int = 0

        self.widget = WorkflowView(self, steps=self.steps)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(500)

    def _register_keybinds(self):
        self.viewer.bind_key("PageDown", self.next_step, overwrite=True)
        self.viewer.bind_key("PageUp", self.prev_step, overwrite=True)
        self.viewer.bind_key("n", self.next_image, overwrite=True)
        self.viewer.bind_key("b", self.prev_image, overwrite=True)
        self.viewer.bind_key("Shift-N", self.next_subject, overwrite=True)
        self.viewer.bind_key("Shift-B", self.prev_subject, overwrite=True)
        self.viewer.bind_key(
            "Home",
            lambda _: self.set_document_index_async(image_index=0),
            overwrite=True,
        )
        self.viewer.bind_key(
            "End",
            lambda _: self.set_document_index_async(
                image_index=len(self.current_subject.valid_documents) - 1
            ),
            overwrite=True,
        )

    def reset(self, save_current_subject: bool = True):
        if self._current_valid_subject_index is not None:
            if save_current_subject:
                self.save_subject()
            self.current_step.close()
            self.widget.set_step(0)
            self.widget.set_image_index(1)
        self.widget.set_subject_index(1)
        self._current_valid_subject_index = None
        self._current_valid_document_index = 0
        self._current_step_index = 0

    def _run_workflow_single_doc(self, doc_i: int) -> None:
        raise NotImplementedError()
        # reader = brainways.utils.io_utils.readers.get_reader(self.documents[doc_i].path)
        # transform = self.current_subject.pipeline.get_image_to_atlas_transform(doc_i, reader)
        # cell_detector_result = None
        # document = self.documents[doc_i]
        #
        # for step in self.steps:
        #     cell_detector_result = step.cells(
        #         reader=reader,
        #         params=document.params,
        #         prev_cell_detector_result=cell_detector_result,
        #     )
        #
        # cells_on_atlas = transform.transform_points(cell_detector_result.cells)
        # self.set_document(replace(document, cells=cells_on_atlas), doc_i)

    def run_workflow_async(self) -> FunctionWorker:
        raise NotImplementedError()
        # self.set_step_index_async(len(self.steps) - 1, run_async=False)
        # view_images = ViewImages(atlas=self._atlas)
        # self.cell_viewer_controller.open(self._atlas)
        # self.widget.show_progress_bar(len(self.documents))
        # worker = create_worker(self._run_workflow)
        # worker.yielded.connect(self._on_run_workflow_yielded)
        # worker.returned.connect(self._on_run_workflow_returned)
        # worker.errored.connect(self._on_work_error)
        # worker.start()
        # return worker

    def _run_workflow(self):
        raise NotImplementedError()
        # for step in self.steps:
        #     step.load_model()
        #
        # for doc_i, doc in enumerate(self.documents):
        #     self._run_workflow_single_doc(doc_i)
        #     yield doc_i

    def _on_run_workflow_yielded(self, doc_index: int):
        raise NotImplementedError()
        # cells = np.concatenate(
        #     [doc.cells for doc in self.documents if doc.cells is not None]
        # )
        # self.cell_viewer_controller.show_cells(cells)
        # self.widget.update_progress_bar(doc_index)

    def _on_run_workflow_returned(self):
        raise NotImplementedError()
        # self.widget.hide_progress_bar()

    def open_project_async(self, path: Path) -> FunctionWorker:
        self.reset()
        return self.do_work_async(
            self._open_project,
            return_callback=self._on_project_opened,
            progress_label="Opening project...",
            path=path,
        )

    def _open_project(self, path: Path):
        yield "Opening project..."
        self.project = BrainwaysProject.open(path, lazy_init=True)
        yield f"Loading '{self.project.settings.atlas}' atlas..."
        self.project.load_atlas()
        yield "Loading Brainways Pipeline models..."
        self.project.load_pipeline()
        if len(self.project.subjects) > 0:
            self._current_valid_subject_index = 0
            yield "Opening image..."
            self._open_image()

    def _open_image(self):
        self._image = self.current_subject.read_lowres_image(self.current_document)
        self._load_step_default_params()

    def _load_step_default_params(self):
        if not self.current_step.has_current_step_params(self.current_params):
            self.current_params = self.current_step.default_params(
                image=self._image, params=self.current_params
            )

    def _open_step(self):
        self.current_step.open()
        self.current_step.show(self.current_params, self._image)
        self.widget.hide_progress_bar()
        self.widget.update_enabled_steps()
        self._set_title()

    def _set_title(self, valid_document_index: Optional[int] = None):
        if valid_document_index is None:
            valid_document_index = self._current_valid_document_index
        _, document = self.current_subject.valid_documents[valid_document_index]
        self.viewer.title = (
            f"{self.current_subject.subject_path.name} - {document.path}"
        )

    def _on_project_opened(self):
        self.widget.on_project_changed(len(self.project.subjects))
        if len(self.project.subjects) > 0:
            self._on_subject_opened()
        self.widget.hide_progress_bar()

    def _on_subject_opened(self):
        self._set_title()
        self._register_keybinds()
        self.widget.on_subject_changed()
        for step in self.steps:
            step.pipeline_loaded()
        self.current_step.open()
        self.current_step.show(self.current_params, self._image)
        self.widget.update_enabled_steps()
        self.widget.hide_progress_bar()

    def _on_progress_returned(self):
        self.widget.hide_progress_bar()

    def persist_current_params(self):
        self.current_document = replace(
            self.current_document, params=self.current_step.params
        )

    def save_subject(self, persist: bool = True) -> None:
        if persist:
            self.persist_current_params()
        self.current_subject.save()
        self.project.save()

    def create_excel_async(
        self,
        path: Path,
        min_region_area_um2: Optional[int] = None,
        cells_per_area_um2: Optional[int] = None,
        min_cell_size_um: Optional[float] = None,
        max_cell_size_um: Optional[float] = None,
        excel_mode: ExcelMode = ExcelMode.ROW_PER_SUBJECT,
    ) -> FunctionWorker:
        return self.do_work_async(
            self.project.create_excel_iter,
            path=path,
            min_region_area_um2=min_region_area_um2,
            cells_per_area_um2=cells_per_area_um2,
            min_cell_size_um=min_cell_size_um,
            max_cell_size_um=max_cell_size_um,
            excel_mode=excel_mode,
            progress_label="Creating Results Excel...",
            progress_max_value=len(self.project.subjects),
        )

    def set_subject_index_async(
        self,
        subject_index: int,
        force: bool = False,
        save_current_subject: bool = True,
    ) -> FunctionWorker | None:
        if not force and self._current_valid_subject_index == subject_index:
            return None
        self.reset(save_current_subject=save_current_subject)
        subject_index = min(max(subject_index, 0), len(self.project.subjects) - 1)

        self._current_valid_subject_index = subject_index
        self.widget.set_subject_index(subject_index + 1)
        self.widget.show_progress_bar()

        worker = create_worker(self._open_image)
        worker.returned.connect(self._on_subject_opened)
        worker.start()
        return worker

    def set_document_index_async(
        self,
        image_index: int,
        force: bool = False,
        persist_current_params: bool = True,
    ) -> FunctionWorker | None:
        if persist_current_params:
            self.save_subject()

        image_index = min(
            max(image_index, 0), len(self.current_subject.valid_documents) - 1
        )
        if not force and self._current_valid_document_index == image_index:
            return None

        self._current_valid_document_index = image_index

        if not self.current_step.enabled(self.current_params):
            self.current_step.close()
            for step_index in reversed(range(self._current_step_index)):
                if self.current_step.enabled(self.current_params):
                    break
                self._current_step_index = step_index
            self.widget.set_step(self._current_step_index)

        self.widget.set_image_index(image_index + 1)
        self.widget.show_progress_bar()

        worker = create_worker(self._open_image)
        worker.returned.connect(self._open_step)
        worker.start()
        return worker

    def prev_subject(self, _=None) -> FunctionWorker | None:
        return self.set_subject_index_async(
            max(self._current_valid_subject_index - 1, 0)
        )

    def next_subject(self, _=None) -> FunctionWorker | None:
        return self.set_subject_index_async(
            min(
                self._current_valid_subject_index + 1,
                len(self.project.subjects) - 1,
            )
        )

    def prev_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            max(self._current_valid_document_index - 1, 0)
        )

    def next_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            min(
                self._current_valid_document_index + 1,
                len(self.current_subject.valid_documents) - 1,
            )
        )

    def set_step_index_async(
        self,
        step_index: int,
        force: bool = False,
        save_subject: bool = True,
        run_async: bool = True,
    ) -> FunctionWorker | None:
        if not force and self._current_step_index == step_index:
            return
        if save_subject:
            self.save_subject()
        self.current_step.close()
        self.widget.set_step(step_index)
        self._current_step_index = step_index
        if run_async:
            worker = create_worker(self._load_step_default_params)
            worker.returned.connect(self._open_step)
            worker.start()
            self.widget.show_progress_bar()
            return worker
        else:
            self._load_step_default_params()
            self._open_step()

    def prev_step(self, _=None) -> FunctionWorker | None:
        return self.set_step_index_async(max(self._current_step_index - 1, 0))

    def next_step(self, _=None) -> FunctionWorker | None:
        return self.set_step_index_async(
            min(self._current_step_index + 1, len(self.steps) - 1)
        )

    def _batch_run_model(self):
        self.widget.show_progress_bar()
        for valid_index in range(len(self.current_subject.valid_documents)):
            self._current_valid_document_index = valid_index
            self._open_image()
            self.current_params = self.current_step.run_model(
                self._image, self.current_params
            )
            self.save_subject(persist=False)
            yield valid_index, self.current_params, self._image

    def _batch_run_model_yielded(
        self, args: Tuple[int, BrainwaysParams, np.ndarray]
    ) -> None:
        valid_index, params, image = args
        self.current_step.show(params, image)
        self.widget.set_image_index(valid_index + 1)
        self.widget.update_progress_bar(valid_index + 1)
        self._set_title(valid_document_index=valid_index)

    def batch_run_model_async(self) -> FunctionWorker:
        self.widget.show_progress_bar(
            max_value=len(self.current_subject.valid_documents)
        )
        worker = create_worker(self._batch_run_model)
        worker.yielded.connect(self._batch_run_model_yielded)
        worker.returned.connect(self._progress_returned)
        worker.start()
        return worker

    def import_cell_detections_async(
        self, path: Path, importer: CellDetectionImporter
    ) -> FunctionWorker:
        return self.do_work_async(
            self.project.import_cell_detections_iter,
            importer=importer,
            cell_detections_root=path,
            progress_label="Importing Cell Detections...",
            progress_max_value=self.project.n_valid_images,
        )

    def run_cell_detector_async(self) -> FunctionWorker:
        return self.do_work_async(
            self.project.run_cell_detector_iter,
            progress_label="Running Cell Detector...",
            progress_max_value=self.project.n_valid_images,
        )

    def show_cells_view(self):
        self.save_subject()
        self.current_step.close()
        self.cell_viewer_controller.open(self.current_subject.atlas)
        cells = np.concatenate(
            [
                doc.cells
                for i, doc in self.current_subject.valid_documents
                if doc.cells is not None
            ]
        )
        self.cell_viewer_controller.show_cells(cells)

    def _progress_returned(self) -> None:
        self.widget.hide_progress_bar()

    def do_work_async(
        self,
        function: Callable,
        return_callback: Optional[Callable] = None,
        yield_callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        progress_label: Optional[str] = None,
        progress_max_value: int = 0,
        **kwargs,
    ) -> FunctionWorker:
        self.widget.show_progress_bar(
            label=progress_label, max_value=progress_max_value
        )
        worker = create_worker(function, **kwargs)
        worker.returned.connect(return_callback or self._on_work_returned)
        if isinstance(worker, GeneratorWorker):
            worker.yielded.connect(yield_callback or self._on_work_yielded)
        worker.errored.connect(error_callback or self._on_work_error)
        worker.start()
        return worker

    def _on_work_returned(self):
        self.widget.hide_progress_bar()

    def _on_work_yielded(self, text: Optional[str] = None, value: Optional[int] = None):
        self.widget.update_progress_bar(value=value, text=text)

    def _on_work_error(self):
        self.widget.hide_progress_bar()

    def _on_progress(self) -> None:
        self.widget.progress_bar.value += 1

    @property
    def _current_document_index(self):
        current_document_index, _ = self.current_subject.valid_documents[
            self._current_valid_document_index
        ]
        return current_document_index

    @property
    def current_valid_document_index(self):
        return self._current_valid_document_index

    @property
    def current_subject(self):
        return self.project.subjects[self._current_valid_subject_index]

    @current_subject.setter
    def current_subject(self, value: BrainwaysSubject):
        assert self._current_valid_subject_index is not None
        self.project.subjects[self._current_valid_subject_index] = value

    @property
    def current_document(self):
        return self.current_subject.documents[self._current_document_index]

    @current_document.setter
    def current_document(self, value: SliceInfo):
        self.current_subject.documents[self._current_document_index] = value

    @property
    def current_step(self):
        return self.steps[self._current_step_index]

    @property
    def current_params(self):
        return self.current_subject.documents[self._current_document_index].params

    @current_params.setter
    def current_params(self, value: BrainwaysParams):
        self.current_document = replace(self.current_document, params=value)

    @property
    def current_subject_index(self):
        return self._current_valid_subject_index

    @property
    def subject_size(self):
        return len(self.current_subject.valid_documents)
