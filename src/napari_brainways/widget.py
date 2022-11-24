from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import napari
import numpy as np
from brainways.pipeline.brainways_params import BrainwaysParams
from brainways.project.brainways_subject import BrainwaysSubject
from brainways.project.info_classes import SliceInfo
from napari.qt.threading import FunctionWorker, GeneratorWorker, create_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari_brainways.controllers.affine_2d_controller import Affine2DController
from napari_brainways.controllers.annotation_viewer_controller import (
    AnnotationViewerController,
)
from napari_brainways.controllers.cell_3d_viewer_controller import (
    Cell3DViewerController,
)
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
        self.cell_viewer_controller = Cell3DViewerController(self)

        self.steps = [
            self.registration_controller,
            self.affine_2d_controller,
            self.tps_controller,
            self.annotation_viewer_controller,
        ]

        self.subject: Optional[BrainwaysSubject] = None
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
        self.viewer.bind_key("Enter", self.next_image, overwrite=True)
        self.viewer.bind_key("Backspace", self.prev_image, overwrite=True)
        self.viewer.bind_key(
            "Home",
            lambda _: self.set_document_index_async(image_index=0),
            overwrite=True,
        )
        self.viewer.bind_key(
            "End",
            lambda _: self.set_document_index_async(
                image_index=len(self.subject.valid_documents) - 1
            ),
            overwrite=True,
        )

    def reset(self):
        if self.subject is not None:
            self.save_subject()
            self.current_step.close()
            self.widget.set_step(0)
            self.widget.set_image_index(1)
        self._current_valid_document_index = 0
        self._current_step_index = 0

    def _load_atlas_async(self) -> FunctionWorker:
        return self.do_work_async(
            self.subject.load_atlas(), progress_label="Loading atlas..."
        )

    def _run_workflow_single_doc(self, doc_i: int) -> None:
        raise NotImplementedError()
        # reader = brainways.utils.io_utils.readers.get_reader(self.documents[doc_i].path)
        # transform = self.subject.pipeline.get_image_to_atlas_transform(doc_i, reader)
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

    def open_subject_async(self, path: Path) -> FunctionWorker:
        self.reset()
        return self.do_work_async(
            self._open_subject,
            return_callback=self._on_subject_opened,
            progress_label="Opening subject...",
            path=path,
        )

    def _open_subject(self, path: Path):
        yield "Opening subject..."
        self.subject = BrainwaysSubject.open(path)
        yield f"Loading '{self.subject.settings.atlas}' atlas..."
        self.subject.load_atlas()
        yield "Loading Brainways Pipeline models..."
        self.subject.load_pipeline()
        yield "Opening image..."
        self._open_image()

    def _open_image(self):
        self._image = self.subject.read_lowres_image(self.current_document)
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
        self._set_title()

    def _set_title(self, valid_document_index: Optional[int] = None):
        if valid_document_index is None:
            valid_document_index = self._current_valid_document_index
        _, document = self.subject.valid_documents[valid_document_index]
        self.viewer.title = f"{self.subject.subject_path.name} - " f"{document.path}"

    def _on_subject_opened(self):
        self._set_title()
        self._register_keybinds()
        self.widget.on_subject_changed()
        for step in self.steps:
            step.pipeline_loaded()
        self.current_step.open()
        self.current_step.show(self.current_params, self._image)
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
        self.subject.save()

    def export_cells(self, path: Path) -> None:
        df = self.subject.cell_count_summary()
        df.to_csv(path, index=False)

    def set_document_index_async(
        self,
        image_index: int,
        force: bool = False,
        persist_current_params: bool = True,
    ) -> FunctionWorker | None:
        if persist_current_params:
            self.save_subject()

        image_index = min(max(image_index, 0), len(self.subject.valid_documents) - 1)
        if not force and self._current_valid_document_index == image_index:
            return None

        self._current_valid_document_index = image_index
        self.widget.set_image_index(image_index + 1)
        self.widget.show_progress_bar()

        worker = create_worker(self._open_image)
        worker.returned.connect(self._open_step)
        worker.start()
        return worker

    def prev_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            max(self._current_valid_document_index - 1, 0)
        )

    def next_image(self, _=None) -> FunctionWorker | None:
        return self.set_document_index_async(
            min(
                self._current_valid_document_index + 1,
                len(self.subject.valid_documents) - 1,
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
        self._current_step_index = step_index
        if run_async:
            worker = create_worker(self._load_step_default_params)
            worker.returned.connect(self._open_step)
            worker.start()
            self.widget.set_step(step_index)
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
        for valid_index in range(len(self.subject.valid_documents)):
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
        self.widget.show_progress_bar(max_value=len(self.subject.valid_documents))
        worker = create_worker(self._batch_run_model)
        worker.yielded.connect(self._batch_run_model_yielded)
        worker.returned.connect(self._progress_returned)
        worker.start()
        return worker

    def _import_cells(self, path: Path) -> None:
        for i, document in self.subject.import_cells_yield_progress(path):
            yield i
        self.subject.save()

    def import_cells_async(self, path: Path) -> FunctionWorker:
        self.widget.show_progress_bar(max_value=len(self.subject.valid_documents))
        worker = create_worker(self._import_cells, path)
        worker.yielded.connect(self._progress_yielded)
        worker.returned.connect(self._progress_returned)
        worker.start()
        return worker

    def show_cells_view(self):
        self.save_subject()
        self.current_step.close()
        self.cell_viewer_controller.open(self.subject.atlas)
        cells = np.concatenate(
            [
                doc.cells
                for i, doc in self.subject.valid_documents
                if doc.cells is not None
            ]
        )
        self.cell_viewer_controller.show_cells(cells)

    def _progress_yielded(self, i: int) -> None:
        raise NotImplementedError()
        # self.widget.set_image_index(i + 1)
        # self.widget.update_progress_bar(i + 1)
        # view_images, params = self._prepare_view_images_and_params(
        #     self.current_document
        # )
        # self.current_step.show(view_images, params)

    def _progress_returned(self) -> None:
        self.widget.hide_progress_bar()

    def do_work_async(
        self,
        function: Callable,
        return_callback: Optional[Callable] = None,
        yield_callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        progress_label: Optional[str] = None,
        **kwargs,
    ) -> FunctionWorker:
        self.widget.show_progress_bar(label=progress_label)
        worker = create_worker(function, **kwargs)
        worker.returned.connect(return_callback or self._on_work_returned)
        if isinstance(worker, GeneratorWorker):
            worker.yielded.connect(yield_callback or self._on_work_yielded)
        worker.errored.connect(error_callback or self._on_work_error)
        worker.start()
        return worker

    def _on_work_returned(self):
        self.widget.hide_progress_bar()

    def _on_work_yielded(self, label: str = "", value: int = 0):
        self.widget.update_progress_bar(value=value, label=label)

    def _on_work_error(self):
        self.widget.hide_progress_bar()

    @property
    def _current_document_index(self):
        current_document_index, _ = self.subject.valid_documents[
            self._current_valid_document_index
        ]
        return current_document_index

    @property
    def current_document(self):
        return self.subject.documents[self._current_document_index]

    @current_document.setter
    def current_document(self, value: SliceInfo):
        self.subject.documents[self._current_document_index] = value

    @property
    def current_step(self):
        return self.steps[self._current_step_index]

    @property
    def current_params(self):
        return self.subject.documents[self._current_document_index].params

    @current_params.setter
    def current_params(self, value: BrainwaysParams):
        self.current_document = replace(self.current_document, params=value)

    @property
    def subject_size(self):
        return len(self.subject.valid_documents)
