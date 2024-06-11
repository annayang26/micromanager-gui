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
from pymmcore_widgets.mda._save_widget import OME_ZARR, WRITERS, ZARR_TESNSORSTORE
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

from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
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
TS = WRITERS[ZARR_TESNSORSTORE][0]
ZR = WRITERS[OME_ZARR][0]


class PlateViewer(QWidget):
    """A widget for displaying a plate preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._datastore: TensorstoreZarrReader | OMEZarrReader | None = None
        self._seg: str | None = None

        # add menu bar
        self.menu_bar = QMenuBar()
        self.file_menu = self.menu_bar.addMenu("File")
        self.file_menu.addAction("Open Zarr Datastore...")
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
        self.splitter_top_left = QSplitter(self, orientation=Qt.Orientation.Vertical)
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
        # self._graph_widget_5 = _GraphWidget(self)
        # self._graph_widget_6 = _GraphWidget(self)
        # self._graph_widget_7 = _GraphWidget(self)
        # self._graph_widget_8 = _GraphWidget(self)
        # self._graph_widget_9 = _GraphWidget(self)
        graphs_layout.addWidget(self._graph_widget_1, 0, 0)
        graphs_layout.addWidget(self._graph_widget_2, 0, 1)
        graphs_layout.addWidget(self._graph_widget_3, 1, 0)
        graphs_layout.addWidget(self._graph_widget_4, 1, 1)
        # graphs_layout.addWidget(self._graph_widget_5, 1, 1)
        # graphs_layout.addWidget(self._graph_widget_6, 1, 2)
        # graphs_layout.addWidget(self._graph_widget_7, 2, 0)
        # graphs_layout.addWidget(self._graph_widget_8, 2, 1)
        # graphs_layout.addWidget(self._graph_widget_9, 2, 2)

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

        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # data = "/Users/fdrgsp/Desktop/test/ts.tensorstore.zarr"
        # reader = TensorstoreZarrReader(data)
        # data = "/Users/fdrgsp/Desktop/test/z.ome.zarr"
        # reader = OMEZarrReader(data)
        # self._seg = "/Users/fdrgsp/Desktop/test/seg"
        # self._init_widget(reader)

    def _set_splitter_sizes(self) -> None:
        """Set the initial sizes for the splitters."""
        splitter_and_sizes = (
            (self.splitter_top_left, [0.73, 0.27]),
            (self.splitter_bottom_left, [0.50, 0.50]),
            (self.main_splitter, [0.30, 0.70]),
        )
        for splitter, sizes in splitter_and_sizes:
            total_size = splitter.size().width()
            splitter.setSizes([int(size * total_size) for size in sizes])

    def _show_init_dialog(self) -> None:
        """Show a dialog to select a zarr datastore file and segmentation path."""
        init_dialog = _InitDialog(
            self,
            datastore_path=(
                str(self._datastore.path) if self._datastore is not None else None
            ),
            segmentation_path=self._seg,
        )
        if init_dialog.exec():
            datastore, self._seg = init_dialog.value()
            # clear fov table
            self._fov_table.clear()
            # clear scene
            self.scene.clear()
            reader: TensorstoreZarrReader | OMEZarrReader
            if datastore.endswith(TS):
                # read tensorstore
                reader = TensorstoreZarrReader(datastore)
            elif datastore.endswith(ZR):
                # read ome zarr
                reader = OMEZarrReader(datastore)
            else:
                show_error_dialog(
                    self,
                    f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE]} and "
                    f"{WRITERS[OME_ZARR]} are supported.",
                )
                return

            self._init_widget(reader)

    def _init_widget(self, reader: TensorstoreZarrReader | OMEZarrReader) -> None:
        """Initialize the widget with the given datastore."""
        self._datastore = reader

        if self._datastore.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot use the  `PlateViewer` without"
                "the useq.MDASequence in the datastore metadata!",
            )
            return

        meta = cast(
            dict, self._datastore.sequence.metadata.get(PYMMCW_METADATA_KEY, {})
        )
        hcs_meta = meta.get(HCS, {})
        if not hcs_meta:
            show_error_dialog(
                self,
                "Cannot open a zarr datastore without HCS metadata! "
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

        if self._datastore is None or value is None:
            return

        if self._datastore.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot retrieve the Well data without "
                "the tensorstore useq.MDASequence!",
            )
            return

        # add the fov per position to the table
        for idx, pos in enumerate(self._datastore.sequence.stage_positions):
            if pos.name and value.name in pos.name:
                self._fov_table.add_position(WellInfo(idx, pos))

        if self._fov_table.rowCount() > 0:
            self._fov_table.selectRow(0)

    def _on_fov_table_selection_changed(self) -> None:
        """Update the image viewer with the first frame of the selected FOV."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None:
            self._image_viewer.setData(None, None)
            self._set_graphs_fov(None)
            return

        if self._datastore is None:
            return

        data = cast(np.ndarray, self._datastore.isel(p=value.idx, t=0, c=0))

        # get one random segmentation between 0 and 2
        seg = self._get_segmentation(value)
        self._image_viewer.setData(data, seg)
        self._set_graphs_fov(value)

    def _set_graphs_fov(self, value: WellInfo | None) -> None:
        """Set the FOV title for the graphs."""
        title = "" if value is None else value.fov.name or f"Position {value.idx}"
        for graph in (
            self._graph_widget_1,
            self._graph_widget_2,
            self._graph_widget_3,
            self._graph_widget_4,
            # self._graph_widget_5,
            # self._graph_widget_6,
            # self._graph_widget_7,
            # self._graph_widget_8,
            # self._graph_widget_9,
        ):
            graph.fov = title

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
        if value is None or self._datastore is None:
            return

        data = self._datastore.isel(p=value.idx)
        viewer = StackViewer(data, parent=self)
        viewer.setWindowTitle(value.fov.name or f"Position {value.idx}")
        viewer.setWindowFlag(Qt.WindowType.Dialog)
        viewer.show()
