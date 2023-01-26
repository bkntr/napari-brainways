import magicgui
from qtpy.QtWidgets import QProgressBar, QVBoxLayout, QWidget


class CellDetectorWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.cell_detector_params_widget = magicgui.magicgui(
            self.cell_detector_params,
            flow_threshold={"min": -100, "max": 100},
            mask_threshold={"min": -100, "max": 100},
            call_button="Run on preview",
        )
        self.cell_detector_params_widget.native.layout().setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.cell_detector_params_widget.native)
        self.layout().addWidget(self.progress_bar)

    def cell_detector_params(
        self,
        diameter: float = 25.0,
        net_avg: bool = True,
        flow_threshold: float = 0.4,
        mask_threshold: float = 0.0,
    ):
        self.controller.run_cell_detector_preview_async(
            diameter=diameter,
            net_avg=net_avg,
            flow_threshold=flow_threshold,
            mask_threshold=mask_threshold,
        )

    def show_progress_bar(self):
        self.cell_detector_params_widget._call_button.enabled = False
        self.progress_bar.show()

    def hide_progress_bar(self):
        self.cell_detector_params_widget._call_button.enabled = True
        self.progress_bar.hide()
