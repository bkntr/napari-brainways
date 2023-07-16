from magicgui.widgets import request_values
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class AnalysisWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.calculate_results_button = QPushButton("Calculate results")
        self.calculate_results_button.clicked.connect(
            self.on_run_calculate_results_clicked
        )

        self.contrast_analysis_button = QPushButton("Run contrast analysis (ANOVA)")
        self.contrast_analysis_button.clicked.connect(
            self.on_run_contrast_analysis_clicked
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.calculate_results_button)
        self.layout().addWidget(self.contrast_analysis_button)

    def on_run_calculate_results_clicked(self, _=None):
        values = request_values(
            title="Excel Parameters",
            min_region_area_um2=dict(
                value=250,
                annotation=int,
                label="Min Structure Square Area (μm)",
                options=dict(
                    tooltip="Filter out structures with an area smaller than this value"
                ),
            ),
            cells_per_area_um2=dict(
                value=250,
                annotation=int,
                label="Cells Per Square Area (μm)",
                options=dict(
                    tooltip="Normalize number of cells to number of cells per area unit"
                ),
            ),
            min_cell_size_um=dict(
                value=0,
                annotation=int,
                label="Min Cell Area (μm)",
                options=dict(
                    tooltip=(
                        "Filter out detected cells with area smaller than this value"
                    )
                ),
            ),
            max_cell_size_um=dict(
                value=0,
                annotation=int,
                label="Max Cell Area (μm)",
                options=dict(
                    tooltip="Filter out detected cells with area larger than this value"
                ),
            ),
        )
        if values is None:
            return

        self.controller.run_calculate_results_async(
            min_region_area_um2=values["min_region_area_um2"],
            cells_per_area_um2=values["cells_per_area_um2"],
            min_cell_size_um=values["min_cell_size_um"],
            max_cell_size_um=values["max_cell_size_um"],
        )

    def on_run_contrast_analysis_clicked(self, _=None):
        conditions = self.controller.ui.project.settings.condition_names
        # cell_types = self.controller.ui.project.cell_types

        values = request_values(
            title="Run Contrast",
            condition_col=dict(
                value=conditions[0],
                widget_type="ComboBox",
                options=dict(choices=conditions),
                annotation=str,
                label="Condition",
            ),
            values_col=dict(
                value="cells",
                # widget_type="ComboBox",
                # options=dict(choices=cell_types),
                annotation=str,
                label="Cell Type",
            ),
            min_group_size=dict(
                value=3,
                annotation=int,
                label="Min Group Size",
                options=dict(
                    tooltip="Minimal number of animals to consider an area for contrast"
                ),
            ),
            pvalue=dict(
                value=0.05,
                annotation=float,
                label="P Value",
                options=dict(tooltip="P value cutoff for posthoc"),
            ),
            multiple_comparisons_method=dict(
                value="fdr_bh",
                annotation=str,
                label="Multiple Comparisons",
                options=dict(
                    tooltip="Method to use when adjusting for multiple comparisons"
                ),
            ),
        )
        if values is None:
            return

        self.controller.run_contrast_analysis(
            condition_col=values["condition_col"],
            values_col=values["values_col"],
            min_group_size=values["min_group_size"],
            pvalue=values["pvalue"],
            multiple_comparisons_method=values["multiple_comparisons_method"],
        )
