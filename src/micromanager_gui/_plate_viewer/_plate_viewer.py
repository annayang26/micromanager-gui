from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import tifffile
from pymmcore_widgets._stack_viewer_v2 import StackViewer
from pymmcore_widgets.hcs._graphics_items import Well, _WellGraphicsItem
from pymmcore_widgets.hcs._plate_model import Plate
from pymmcore_widgets.hcs._util import _ResizingGraphicsView, draw_plate
from pymmcore_widgets.mda._core_mda import HCS
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPen
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QMenuBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

from ._fov_table import WellInfo, _FOVTable
from ._graph_widget import _GraphWidget
from ._image_viewer import _ImageViewer
from ._init_dialog import _InitDialog
from ._util import show_error_dialog
from ._wells_graphic_scene import _WellsGraphicsScene

GREEN = "#00FF00"  # "#00C600"
SELECTED_COLOR = QBrush(QColor(GREEN))
UNSELECTED_COLOR = QBrush(Qt.GlobalColor.lightGray)
UNSELECTABLE_COLOR = QBrush(Qt.GlobalColor.darkGray)
PEN = QPen(Qt.GlobalColor.black)
PEN.setWidth(3)
OPACITY = 0.7


class PlateViewer(QWidget):
    """A widget for displaying a plate preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._ts: TensorstoreZarrReader | None = None
        self._seg: str | None = None

        # add menu bar
        self.menu_bar = QMenuBar()
        self.file_menu = self.menu_bar.addMenu("File")
        self.file_menu.addAction("Open Zarr Tensorstore")
        self.file_menu.triggered.connect(self._show_init_dialog)

        # scene and view for the plate map
        self.scene = _WellsGraphicsScene()
        self.view = _ResizingGraphicsView(self.scene)
        self.view.setStyleSheet("background:grey; border-radius: 5px;")

        # table for the fields of view
        self._fov_table = _FOVTable(self)
        self._fov_table.itemSelectionChanged.connect(
            self._on_fov_table_selection_changed
        )
        self._fov_table.doubleClicked.connect(self._on_fov_double_click)

        # image viewer
        self._image_viewer = _ImageViewer(self)

        # splitter for the plate map and the fov table
        self.splitter_top_left = QSplitter(self)
        self.splitter_top_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_top_left.setChildrenCollapsible(False)
        self.splitter_top_left.addWidget(self.view)
        self.splitter_top_left.addWidget(self._fov_table)
        # splitter for the plate map/fov table and the image viewer
        self.splitter_bottom_left = QSplitter(self, orientation=Qt.Orientation.Vertical)
        self.splitter_bottom_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_bottom_left.setChildrenCollapsible(False)
        self.splitter_bottom_left.addWidget(self.splitter_top_left)
        self.splitter_bottom_left.addWidget(self._image_viewer)
        # graphs widget
        graphs_wdg = QGroupBox()
        graphs_layout = QGridLayout(graphs_wdg)
        self._graph_widget_1 = _GraphWidget(self)
        self._graph_widget_2 = _GraphWidget(self)
        self._graph_widget_3 = _GraphWidget(self)
        self._graph_widget_4 = _GraphWidget(self)
        graphs_layout.addWidget(self._graph_widget_1, 0, 0)
        graphs_layout.addWidget(self._graph_widget_2, 0, 1)
        graphs_layout.addWidget(self._graph_widget_3, 1, 0)
        graphs_layout.addWidget(self._graph_widget_4, 1, 1)

        # splitter between the plate map/fov table/image viewer and the graphs
        self.main_splitter = QSplitter(self)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter_bottom_left)
        self.main_splitter.addWidget(graphs_wdg)

        # add widgets to the layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.addWidget(self.main_splitter)

        self.scene.selectedWellChanged.connect(self._on_scene_well_changed)

        self.showMaximized()

        self._set_init_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        self._read_tensorstore("/Users/fdrgsp/Desktop/test/ts.tensorstore.zarr")
        self._seg = "/Users/fdrgsp/Desktop/test/seg"

    def _set_init_splitter_sizes(self) -> None:
        """Set the initial sizes for the splitters."""
        splitter_and_sizes = (
            (self.splitter_top_left, [0.75, 0.25]),
            (self.splitter_bottom_left, [0.35, 0.65]),
            (self.main_splitter, [0.40, 0.60]),
        )
        for splitter, sizes in splitter_and_sizes:
            total_size = splitter.size().width()
            splitter.setSizes([int(size * total_size) for size in sizes])

    def _show_init_dialog(self) -> None:
        """Show a dialog to select tensorstore.zarr file and segmentation path."""
        init_dialog = _InitDialog(
            self,
            tensorstore_path=str(self._ts.path) if self._ts is not None else None,
            segmentation_path=self._seg,
        )
        if init_dialog.exec():
            ts, self._seg = init_dialog.value()
            # clear fov table
            self._fov_table.clear()
            # clear scene
            self.scene.clear()
            # read tensorstore
            self._read_tensorstore(ts)

    def _read_tensorstore(self, ts: str) -> None:
        """Read the tensorstore.zarr and populate the plate viewer."""
        self._ts = TensorstoreZarrReader(ts)

        if self._ts.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot use the  `PlateViewer` without"
                "the tensorstore useq.MDASequence!",
            )
            return

        meta = cast(dict, self._ts.sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        hcs_meta = meta.get(HCS, {})
        if not hcs_meta:
            show_error_dialog(
                self,
                "Cannot open a tensorstore.zarr without HCS metadata! "
                f"Metadata: {meta}",
            )
            return

        plate = hcs_meta.get("plate")
        if not plate:
            show_error_dialog(
                self,
                "Cannot find plate information in the HCS metadata! "
                f"HCS Metadata: {hcs_meta}",
            )
            return
        plate = plate if isinstance(plate, Plate) else Plate(**plate)

        # draw plate
        draw_plate(self.view, self.scene, plate, UNSELECTED_COLOR, PEN, OPACITY)

        # get acquired wells (use row and column and not the name to be safer)
        wells_row_col = []
        for well in hcs_meta.get("wells", []):
            well = well if isinstance(well, Well) else Well(**well)
            wells_row_col.append((well.row, well.column))

        # disable non-acquired wells
        to_exclude = []
        for item in self.scene.items():
            item = cast(_WellGraphicsItem, item)
            well = item.value()
            if (well.row, well.column) not in wells_row_col:
                item.brush = UNSELECTABLE_COLOR
                to_exclude.append(item.value())
        self.scene.exclude_wells = to_exclude

    def _on_scene_well_changed(self, value: Well | None) -> None:
        """Update the FOV table when a well is selected."""
        self._fov_table.clear()

        if self._ts is None or value is None:
            return

        if self._ts.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot retrieve the Well data without "
                "the tensorstore useq.MDASequence!",
            )
            return

        # add the fov per position to the table
        for idx, pos in enumerate(self._ts.sequence.stage_positions):
            if pos.name and value.name in pos.name:
                self._fov_table.add_position(WellInfo(idx, pos))

        if self._fov_table.rowCount() > 0:
            self._fov_table.selectRow(0)

    def _on_fov_table_selection_changed(self) -> None:
        """Update the image viewer with the first frame of the selected FOV."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None:
            self._image_viewer.setData(None, None)
            return

        if self._ts is None:
            return

        data = cast(np.ndarray, self._ts.isel(p=value.idx, t=0, c=0))
        # get one random segmentation between 0 and 2
        seg = self._get_segmentation(value)
        self._image_viewer.setData(data, seg)

    def _get_segmentation(self, value: WellInfo) -> np.ndarray | None:
        """Get the segmentation for the given FOV."""
        if self._seg is None:
            return None
        # the segmentation tif file should have the same name as the position
        # and should end with _on where n is the position number (e.g. C3_0000_p0.tif)
        pos_idx = f"p{value.idx}"
        pos_name = value.fov.name
        for f in Path(self._seg).iterdir():
            name = f.name.replace(f.suffix, "")
            if pos_name and pos_name in f.name and name.endswith(f"_{pos_idx}"):
                return tifffile.imread(f)  # type: ignore
        return None

    def _on_fov_double_click(self) -> None:
        """Open the selected FOV in a new StackViewer window."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None or self._ts is None:
            return

        data = self._ts.isel(p=value.idx)
        viewer = StackViewer(data, parent=self)
        viewer.setWindowTitle(value.fov.name or f"Position {value.idx}")
        viewer.setWindowFlag(Qt.WindowType.Dialog)
        viewer.show()
