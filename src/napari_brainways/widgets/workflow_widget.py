import functools
import os
from pathlib import Path
from typing import Callable, List, Union

import importlib_resources
import PIL.Image
from bg_atlasapi.list_atlases import get_all_atlases_lastversions
from brainways.project.brainways_project import BrainwaysProject
from brainways.project.info_classes import ExcelMode, ProjectSettings
from brainways.utils.cell_detection_importer.utils import (
    cell_detection_importer_types,
    get_cell_detection_importer,
)
from magicgui import magicgui
from magicgui.widgets import Container, Image, Label, PushButton, Widget, request_values
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from napari_brainways.controllers.base import Controller
from napari_brainways.widgets.create_subject_dialog import CreateSubjectDialog


class WorkflowView(QWidget):
    def __init__(self, controller, steps: List[Controller]):
        super().__init__(controller)
        self.controller = controller
        self.steps = steps
        self._prev_path = str(Path.home())

        self.project_buttons = ProjectButtons(
            open_project=self.on_open_project_clicked,
            edit_project=self.on_edit_subject_clicked,
            new_project=self.on_create_project_clicked,
        )
        self.project_actions_section = ProjectActionsSection(
            export_excel=self.on_export_clicked,
            import_cells=self.on_import_cells_clicked,
        )
        self.subject_navigation = SubjectControls(
            select_callback=self.select_subject,
            prev_callback=self.controller.prev_subject,
            next_callback=self.controller.next_subject,
            add_subject_callback=self.on_add_subject_clicked,
        )
        self.image_navigation = NavigationControls(
            title="<b>Image Controls:</b> [b/n]",
            label="Image",
            select_callback=self.select_image,
            prev_callback=self.controller.prev_image,
            next_callback=self.controller.next_image,
        )
        self.step_buttons = StepButtons(
            steps=steps, clicked=self.on_step_clicked, title="<b>Steps:</b> [PgUp/PgDn]"
        )
        self.step_controls = StepControls(steps=steps)
        self.progress_bar = ProgressBar()
        self.header_section = HeaderSection(progress_bar=self.progress_bar)
        self.subject_controls = self._stack_widgets(
            [
                self.image_navigation,
                self.step_buttons,
                self.step_controls,
                self.subject_navigation,
                self.project_actions_section,
            ]
        )
        self.subject_controls.hide()

        layout = self._stack_widgets(
            [
                self.header_section,
                self.project_buttons,
                self.subject_controls,
            ]
        )
        self.setLayout(layout.layout())

        self.setMinimumWidth(400)

    def _stack_widgets(self, widgets) -> QWidget:
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().setContentsMargins(0, 0, 0, 0)
        for section in widgets:
            widget.layout().addWidget(section)
        return widget

    def on_create_project_clicked(self, _=None):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "New Brainways Project",
            self._prev_path,
            "Brainways Project File (*.bwp)",
        )
        if path == "":
            return
        self._prev_path = str(Path(path).parent)

        available_atlases = list(get_all_atlases_lastversions().keys())
        atlas = request_values(
            title="New Brainways Project",
            atlas=dict(
                value="whs_sd_rat_39um",
                widget_type="ComboBox",
                options=dict(choices=available_atlases),
                annotation=str,
                label="Importer Type",
            ),
        )["atlas"]

        settings = ProjectSettings(atlas=atlas, channel=0)
        project = BrainwaysProject.create(path=path, settings=settings, lazy_init=True)
        dialog = CreateSubjectDialog(project=project, parent=self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return
        self.controller.open_project_async(path)

    def on_add_subject_clicked(self, _=None):
        path = QFileDialog.getExistingDirectory(
            self,
            "Create Brainways Project",
            str(self._prev_path),
        )
        if path == "":
            return
        self._prev_path = str(Path(path))
        dialog = CreateSubjectDialog(project=self.controller.project, parent=self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return
        self.controller.open_subject_async(dialog.subject.subject_path)

    def on_edit_subject_clicked(self, _=None):
        dialog = CreateSubjectDialog(
            project=self.controller.project,
            subject_index=self.controller.current_subject_index,
            parent=self,
        )
        result = dialog.exec()
        if result == QDialog.DialogCode.Rejected:
            return
        self.on_subject_changed()
        self.controller.set_document_index_async(
            image_index=0, force=True, persist_current_params=False
        )

    def on_open_project_clicked(self, _=None):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._prev_path,
            "Brainways Project (*.bwp)",
            **kwargs,
        )
        if path == "":
            return
        self._prev_path = str(Path(path).parent)
        self.controller.open_project_async(Path(path))

    def on_project_changed(self, n_subjects: int):
        if n_subjects == 0:
            self.subject_navigation.visible = False
        else:
            self.project_buttons.project_opened()
            self.subject_navigation.max = n_subjects
            self.subject_navigation.visible = True
            self.set_step(0)

    def on_subject_changed(self):
        self.image_navigation.max = self.controller.subject_size
        self.subject_controls.show()

    def select_subject(self, value: int):
        self.controller.set_subject_index_async(value - 1)

    def select_image(self, value: int):
        self.controller.set_document_index_async(value - 1)

    def set_step(self, step_index: int):
        self.step_buttons.set_step(step_index)
        self.step_controls.set_step(step_index)

    def on_step_clicked(self, _=None, step_index: int = 0):
        self.set_step(step_index)
        self.controller.set_step_index_async(step_index)

    def on_prev_step_clicked(self, _=None):
        self.controller.prev_step()

    def on_next_step_clicked(self, _=None):
        self.controller.next_step()

    def on_run_workflow_clicked(self, _=None):
        self.controller.run_workflow_async()

    def on_save_button_clicked(self, _=None):
        self.controller.save_subject()

    def on_export_clicked(self, _=None):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            self._prev_path,
            "XLSX File (*.xlsx)",
            **kwargs,
        )
        if path == "":
            return
        self._prev_path = str(Path(path).parent)

        values = request_values(
            title="Excel Parameters",
            excel_mode=dict(
                value=ExcelMode.ROW_PER_SUBJECT,
                annotation=ExcelMode,
                label="Excel Mode",
                options=dict(
                    tooltip="Output a row per subject or row per image (useful for "
                    "error analysis)"
                ),
            ),
            min_region_area_um2=dict(
                value=250,
                annotation=int,
                label="Min Structure Area (μm²)",
                options=dict(
                    tooltip="Filter out structures with an area smaller than this value"
                ),
            ),
            cells_per_area_um2=dict(
                value=250,
                annotation=int,
                label="Cells Per Area (μm²)",
                options=dict(
                    tooltip="Normalize number of cells to number of cells per area unit"
                ),
            ),
        )

        self.controller.create_excel_async(
            Path(path),
            min_region_area_um2=values["min_region_area_um2"],
            cells_per_area_um2=values["cells_per_area_um2"],
            excel_mode=values["excel_mode"],
        )

    def on_import_cells_clicked(self, _=None):
        kwargs = {}
        if "SNAP" in os.environ:
            kwargs["options"] = QFileDialog.DontUseNativeDialog

        path = QFileDialog.getExistingDirectory(
            self,
            "Import Cells",
            self._prev_path,
            **kwargs,
        )
        if path == "":
            return
        self._prev_path = str(Path(path))

        importer_type = request_values(
            title="Import Cell Detections",
            importer_type=dict(
                value="keren",
                widget_type="ComboBox",
                options=dict(choices=cell_detection_importer_types()),
                annotation=str,
                label="Importer Type",
            ),
        )["importer_type"]

        Importer = get_cell_detection_importer(importer_type)
        importer_params = {}
        if Importer.parameters:
            importer_params = request_values(
                title="Import Cell Detections Parameters", values=Importer.parameters
            )

        self.controller.import_cell_detections_async(
            path=Path(path), importer=Importer(**importer_params)
        )

    def set_subject_index(self, subject_index: int):
        self.subject_navigation.value = subject_index

    def set_image_index(self, image_index: int):
        self.image_navigation.value = image_index

    def update_progress_bar(self, value: int = None, text: str = None):
        if text is not None:
            self.progress_bar.text = text
        if value is not None:
            self.progress_bar.value = value
        elif value is None and self.progress_bar.max > 0:
            self.progress_bar.value += 1

    def show_progress_bar(self, max_value: int = 0, label: str = ""):
        self.setEnabled(False)
        self.progress_bar.text = label
        self.progress_bar.max = max_value
        self.header_section.show_progress()

    def hide_progress_bar(self):
        self.setEnabled(True)
        self.header_section.hide_progress()


class TitledGroupBox(QWidget):
    def __init__(
        self,
        title: Union[str, QLabel],
        widgets: List[Union[QWidget, Widget]],
        layout: str = "vertical",
        visible: bool = True,
    ):
        super().__init__()
        groupbox = QGroupBox()
        if layout == "vertical":
            groupbox.setLayout(QVBoxLayout())
        else:
            groupbox.setLayout(QHBoxLayout())

        for widget in widgets:
            if isinstance(widget, Widget):
                widget = widget.native
            groupbox.layout().addWidget(widget)
            groupbox.layout().addWidget(widget)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel(title) if isinstance(title, str) else title)
        self.layout().addWidget(groupbox)

        self.visible = visible

    @property
    def visible(self) -> bool:
        return self.isVisible()

    @visible.setter
    def visible(self, value: bool):
        self.setVisible(value)


class ProjectButtons(TitledGroupBox):
    def __init__(
        self,
        open_project: Callable,
        edit_project: Callable,
        new_project: Callable,
    ):
        self.open_project = PushButton(text="Open")
        self.edit_project = PushButton(text="Edit")
        self.new_project = PushButton(text="New")

        self.open_project.clicked.connect(open_project)
        self.edit_project.clicked.connect(edit_project)
        self.new_project.clicked.connect(new_project)

        super().__init__(
            title="<b>Project:</b>",
            widgets=[self.open_project, self.edit_project, self.new_project],
            layout="horizontal",
        )

        self.project_closed()

    def project_opened(self):
        self.edit_project.visible = True

    def project_closed(self):
        self.edit_project.visible = False


class ProjectActionsSection(TitledGroupBox):
    def __init__(
        self,
        export_excel: Callable,
        import_cells: Callable,
    ):
        self.export_excel = PushButton(text="Create Results Excel")
        self.import_cells = PushButton(text="Import Cell Detections")

        self.export_excel.clicked.connect(export_excel)
        self.import_cells.clicked.connect(import_cells)

        super().__init__(
            title="<b>Project Actions:</b>",
            widgets=[self.export_excel, self.import_cells],
        )


class NavigationControls(TitledGroupBox):
    def __init__(
        self,
        title: str,
        label: str,
        select_callback: Callable,
        prev_callback: Callable,
        next_callback: Callable,
        visible: bool = True,
    ):
        self.selector_widget = magicgui(
            select_callback,
            auto_call=True,
            value={
                "widget_type": "Slider",
                "label": f"{label} #",
                "min": 1,
                "max": 1,
            },
        )
        self.selector_max_label = Label(value="")
        self.prev_button = PushButton(text="< Previous")
        self.next_button = PushButton(text="Next >")

        self.prev_button.clicked.connect(prev_callback)
        self.next_button.clicked.connect(next_callback)

        super().__init__(title=title, widgets=self._build_layout(), visible=visible)

    def _build_layout(self) -> List[Widget]:
        self.selector_widget.native.layout().setContentsMargins(0, 0, 0, 0)
        selector = Container(
            widgets=[self.selector_widget, self.selector_max_label],
            layout="horizontal",
            labels=False,
        )
        buttons = Container(
            widgets=[self.prev_button, self.next_button],
            layout="horizontal",
            labels=False,
        )

        selector.margins = (0, 0, 0, 0)
        buttons.margins = (0, 0, 0, 0)

        return [selector, buttons]

    @property
    def max(self):
        return self.selector_widget.value.max

    @max.setter
    def max(self, value: int):
        self.selector_widget.value.max = value
        self.selector_max_label.value = f"/ {value}"

    @property
    def visible(self) -> bool:
        return self.isVisible()

    @visible.setter
    def visible(self, value: bool):
        self.setVisible(value)

    @property
    def value(self):
        return self.selector_widget.value

    @value.setter
    def value(self, value):
        self.selector_widget._auto_call = False
        self.selector_widget.value.value = value
        self.selector_widget._auto_call = True


class SubjectControls(NavigationControls):
    def __init__(
        self,
        select_callback: Callable,
        prev_callback: Callable,
        next_callback: Callable,
        add_subject_callback: Callable,
        visible: bool = True,
    ):
        self.add_subject_button = PushButton(text="Add Subject")
        self.add_subject_button.clicked.connect(add_subject_callback)

        super().__init__(
            title="<b>Subject Controls:</b> [B/N]",
            label="Subject",
            select_callback=select_callback,
            prev_callback=prev_callback,
            next_callback=next_callback,
            visible=visible,
        )

    def _build_layout(self) -> List[Widget]:
        widgets = super()._build_layout()
        widgets.append(self.add_subject_button)
        return widgets


class StepButtons(TitledGroupBox):
    def __init__(self, steps: List[Controller], clicked: Callable, title: str):
        self.buttons = []
        for i, step in enumerate(steps):
            button = PushButton(text=step.name)
            button.clicked.connect(functools.partial(clicked, step_index=i))
            button.clicked.connect(functools.partial(self.set_step, step_index=i))
            button.native.setCheckable(True)
            self.buttons.append(button)

        super().__init__(title=title, widgets=self.buttons)

    def set_step(self, step_index: int):
        for i, button in enumerate(self.buttons):
            button.native.setChecked(i == step_index)


class StepControls(TitledGroupBox):
    def __init__(self, steps: List[Controller]):
        self.steps = steps
        self.title = QLabel("")
        font = self.title.font()
        font.setPointSize(11)
        self.title.setFont(font)
        self.widgets = [step.widget for step in steps if step.widget is not None]
        super().__init__(title=self.title, widgets=self.widgets)

    def set_step(self, step_index: int):
        current_step = self.steps[step_index]
        self.setVisible(current_step.widget is not None)
        self.title.setText(f"<b>{current_step.name} Parameters:</b>")


class ProgressBar(QWidget):
    def __init__(self):
        super().__init__()
        self._label_widget = QLabel(self)
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(0)

        self.setLayout(QVBoxLayout(self))
        self.layout().addWidget(self._label_widget)
        self.layout().addWidget(self._progress_bar)

        self.hide()

    @property
    def max(self) -> int:
        return self._progress_bar.maximum()

    @max.setter
    def max(self, value: int):
        self._progress_bar.setMaximum(value)

    @property
    def value(self) -> int:
        return self._progress_bar.value()

    @value.setter
    def value(self, value: int):
        self._progress_bar.setValue(value)

    @property
    def text(self) -> str:
        return self._label_widget.text()

    @text.setter
    def text(self, value: str):
        self._label_widget.setText(value)


class HeaderSection(QWidget):
    def __init__(self, progress_bar: ProgressBar):
        super().__init__()

        self.header = self._build_header()
        self.progress_bar = progress_bar
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.header)
        self.layout().addWidget(progress_bar)
        self.setMinimumHeight(80)

    def _build_header(self):
        package = Path(importlib_resources.files("napari_brainways"))
        logo = Image(value=PIL.Image.open(package / "resources/logo.png"))
        title = QLabel("Brainways")
        font = title.font()
        font.setPointSize(16)
        title.setFont(font)

        header_container = QWidget()
        header_container.setLayout(QHBoxLayout())
        header_container.layout().addWidget(logo.native)
        header_container.layout().addWidget(title)

        # header_container.layout().setAlignment(title, Qt.AlignLeft)
        return header_container

    def show_progress(self):
        self.header.hide()
        self.progress_bar.show()

    def hide_progress(self):
        self.header.show()
        self.progress_bar.hide()
