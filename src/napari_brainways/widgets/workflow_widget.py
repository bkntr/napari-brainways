import functools
import os
from pathlib import Path
from typing import List

from brainways.utils.paths import ANNOTATE_V1_1_ROOT
from magicgui import magicgui
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_brainways.controllers.base import Controller
from napari_brainways.widgets.create_project_dialog import CreateProjectDialog


class WorkflowView(QWidget):
    def __init__(self, controller, steps: List[Controller]):
        super().__init__(controller)
        self.controller = controller
        self.steps = steps
        self._project_init_widget = self._create_project_init_buttons()

        (
            self._image_controls_groupbox,
            self._select_image_widget,
            self._select_image_label,
        ) = self._create_image_navigation_controls()

        (self.steps_groupbox, self._step_buttons) = self._create_step_buttons()
        (
            self._step_controls_widget,
            self._step_controls_header,
        ) = self._create_step_controls()
        self.set_step(0)

        (
            self._progress_bar_layout_widget,
            self._progress_bar,
            self._progress_bar_label,
        ) = self._create_progress_bar()

        self.cell_view_button = QPushButton("3D Cell View")
        self.cell_view_button.clicked.connect(self.controller.show_cells_view)

        self.edit_project_button = QPushButton("Edit Project")
        self.edit_project_button.clicked.connect(self.on_edit_project_clicked)

        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.on_save_button_clicked)

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.on_export_clicked)

        self.run_workflow_button = QPushButton("Run Workflow")
        self.run_workflow_button.clicked.connect(self.on_run_workflow_clicked)

        self.batch_run_model_button = QPushButton("Batch Run Step")
        self.batch_run_model_button.clicked.connect(
            self.controller.batch_run_model_async
        )

        self.import_cells_button = QPushButton("Import Cells")
        self.import_cells_button.clicked.connect(self.on_import_cells_clicked)

        self._project_controls_widget = QWidget()
        self._project_controls_layout = QVBoxLayout(self._project_controls_widget)

        self._project_controls_layout.addWidget(
            QLabel("<b>Image Controls:</b> [Enter/Backspace]")
        )
        self._project_controls_layout.addWidget(self._image_controls_groupbox)
        self._project_controls_layout.addWidget(QLabel("<b>Steps:</b> [PgUp/PgDn]"))
        self._project_controls_layout.addWidget(self.steps_groupbox)
        self._project_controls_layout.addWidget(self._step_controls_widget)
        self._project_controls_layout.addWidget(self.cell_view_button)
        self._project_controls_layout.addWidget(self.import_cells_button)
        self._project_controls_layout.addWidget(self.batch_run_model_button)
        self._project_controls_layout.addWidget(self.run_workflow_button)
        self._project_controls_layout.addWidget(self.save_button)
        self._project_controls_layout.addWidget(self.edit_project_button)
        self._project_controls_layout.addWidget(self.export_button)
        self._project_controls_widget.hide()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.addWidget(self._project_init_widget)
        self._layout.addWidget(self._project_controls_widget)
        self._layout.addWidget(self._progress_bar_layout_widget)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.setMinimumWidth(500)

    def _create_progress_bar(self):
        progress_bar_layout_widget = QWidget()
        progress_bar_layout_widget.hide()
        progress_bar_layout = QVBoxLayout(progress_bar_layout_widget)
        progress_bar_label = QLabel(progress_bar_layout_widget)
        progress_bar = QProgressBar(progress_bar_layout_widget)
        progress_bar.setValue(0)
        progress_bar.setMaximum(0)
        progress_bar_layout.addWidget(progress_bar_label)
        progress_bar_layout.addWidget(progress_bar)

        return progress_bar_layout_widget, progress_bar, progress_bar_label

    def _create_project_init_buttons(self):
        create_project_button = QPushButton("Create Project", self)
        create_project_button.clicked.connect(self.on_create_project_clicked)
        open_project_button = QPushButton("Open Project", self)
        open_project_button.clicked.connect(self.on_open_project_clicked)
        project_init_widget = QWidget()
        project_init_layout = QVBoxLayout(project_init_widget)
        project_init_layout.addWidget(create_project_button)
        project_init_layout.addWidget(open_project_button)
        return project_init_widget

    def _create_image_navigation_controls(self):
        select_image_widget = magicgui(
            self.select_image,
            auto_call=True,
            image_index={
                "widget_type": "Slider",
                "label": "Image #",
                "min": 1,
                "max": 1,
            },
        )
        select_image_widget.native.layout().setContentsMargins(0, 0, 0, 0)
        select_image_label = QLabel("", self)
        select_image_layout = QHBoxLayout()
        select_image_layout.addWidget(select_image_widget.native)
        select_image_layout.addWidget(select_image_label)
        image_controls_groupbox = QGroupBox()
        image_controls_layout = QVBoxLayout()
        image_controls_groupbox.setLayout(image_controls_layout)
        prev_image_button = QPushButton("< Previous")
        prev_image_button.clicked.connect(self.on_prev_image_clicked)
        next_image_button = QPushButton("Next >")
        next_image_button.clicked.connect(self.on_next_image_clicked)
        prev_next_image_buttons = QHBoxLayout()
        prev_next_image_buttons.addWidget(prev_image_button)
        prev_next_image_buttons.addWidget(next_image_button)
        image_controls_layout.addLayout(select_image_layout)
        image_controls_layout.addLayout(prev_next_image_buttons)

        return image_controls_groupbox, select_image_widget, select_image_label

    def _create_step_buttons(self):
        groupbox = QGroupBox()
        layout = QVBoxLayout()
        groupbox.setLayout(layout)
        step_buttons = []
        for i, controller in enumerate(self.steps):
            step_button = QPushButton(controller.name)
            step_button.clicked.connect(
                functools.partial(self.on_step_clicked, step_index=i)
            )
            step_button.setCheckable(True)
            step_buttons.append(step_button)
            layout.addWidget(step_button)
        return groupbox, step_buttons

    def _create_step_controls(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        header = QLabel("")
        header.font().setPointSize(11)
        groupbox = QGroupBox()
        layout.addWidget(header)
        layout.addWidget(groupbox)
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(0, 0, 0, 0)
        groupbox.setLayout(inner_layout)
        for i, controller in enumerate(self.steps):
            if controller.widget is not None:
                inner_layout.addWidget(controller.widget)
        return widget, header

    def on_create_project_clicked(self, _):
        dialog = CreateProjectDialog(self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return
        self.controller.open_project_async(dialog.project.project_path)

    def on_edit_project_clicked(self, _):
        dialog = CreateProjectDialog(
            self,
            project=self.controller.project,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return
        self.controller.project.documents = dialog.documents
        self.controller.set_document_index_async(
            image_index=0, force=True, persist_current_params=False
        )
        self.on_project_changed()

    def on_open_project_clicked(self, _):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(ANNOTATE_V1_1_ROOT),
            "Brainways project (*.bin)",
            **kwargs,
        )
        if path == "":
            return
        self.controller.open_project_async(Path(path))

    def on_project_changed(self):
        self._select_image_widget.image_index.max = self.controller.project_size
        self._select_image_label.setText(
            f"/ {self._select_image_widget.image_index.max}"
        )
        self._project_controls_widget.show()

    def select_image(self, image_index: int):
        self.controller.set_document_index_async(image_index - 1)

    def on_prev_image_clicked(self, _):
        self.controller.prev_image()

    def on_next_image_clicked(self, _):
        self.controller.next_image()

    def set_step(self, step_index: int):
        for step_button in self._step_buttons:
            step_button.setChecked(False)
        current_step = self.steps[step_index]
        self._step_controls_widget.setVisible(current_step.widget is not None)
        self._step_buttons[step_index].setChecked(True)
        self._step_controls_header.setText(f"<b>{current_step.name} Parameters:</b>")

    def on_step_clicked(self, _, step_index: int):
        self.set_step(step_index)
        self.controller.set_step_index_async(step_index)

    def on_prev_step_clicked(self, _):
        self.controller.prev_step()

    def on_next_step_clicked(self, _):
        self.controller.next_step()

    def on_run_workflow_clicked(self, _):
        self.controller.run_workflow_async()

    def on_save_button_clicked(self, _):
        self.controller.save_project()

    def on_export_clicked(self, _):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            str(Path.home()),
            "CSV File (*.csv)",
            **kwargs,
        )
        self.controller.export_cells(Path(path))

    def on_import_cells_clicked(self, _):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path = QFileDialog.getExistingDirectory(
            self,
            "Import Cells",
            str(Path.home()),
            **kwargs,
        )
        if path == "":
            return
        self.controller.import_cells_async(Path(path))

    def set_image_index(self, image_index: int):
        self._select_image_widget._auto_call = False
        self._select_image_widget.image_index.value = image_index
        self._select_image_widget._auto_call = True
        self._select_image_label.setText(
            f"/ {self._select_image_widget.image_index.max}"
        )

    def update_progress_bar(self, value: int, label: str = None):
        if label is not None:
            self._progress_bar_label.setText(label)
        self._progress_bar.setValue(value)

    def show_progress_bar(self, max_value: int = 0, label: str = ""):
        self.setEnabled(False)
        self._progress_bar_label.setText(label)
        self._progress_bar.setMaximum(max_value)
        self._progress_bar_layout_widget.show()

    def hide_progress_bar(self):
        self.setEnabled(True)
        self._progress_bar_layout_widget.hide()
