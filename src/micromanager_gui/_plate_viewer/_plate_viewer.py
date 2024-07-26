from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator, cast

import numpy as np
import tifffile
from fonticon_mdi6 import MDI6
from pymmcore_widgets._stack_viewer_v2 import StackViewer
from pymmcore_widgets.hcs._graphics_items import Well, _WellGraphicsItem
from pymmcore_widgets.hcs._plate_model import Plate
from pymmcore_widgets.hcs._util import _ResizingGraphicsView, draw_plate
from pymmcore_widgets.mda._core_mda import HCS
from pymmcore_widgets.mda._save_widget import OME_ZARR, WRITERS, ZARR_TESNSORSTORE
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QBrush, QColor, QIcon, QPen
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

from ._analysis import _AnalyseCalciumTraces
from ._fov_table import WellInfo, _FOVTable
from ._graph_widget import _GraphWidget
from ._image_viewer import _ImageViewer
from ._init_dialog import _InitDialog
from ._plate_map import PlateMapWidget
from ._segmentation import _CellposeSegmentation
from ._util import (
    GENOTYPE_MAP,
    TREATMENT_MAP,
    Peaks,
    ROIData,
    _ProgressBarWidget,
    show_error_dialog,
)
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


class PlateViewer(QMainWindow):
    """A widget for displaying a plate preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        labels_path: str | None = None,
        analysis_file_path: str | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Plate Viewer")
        self.setWindowIcon(QIcon(icon(MDI6.view_comfy, color="#00FF00")))

        # add central widget
        self._central_widget = QWidget(self)
        self._central_widget_layout = QVBoxLayout(self._central_widget)
        self._central_widget_layout.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(self._central_widget)

        self._datastore: TensorstoreZarrReader | OMEZarrReader | None = None
        self._labels_path = labels_path
        self._analysis_file_path = analysis_file_path

        # maybe make it as a pandas dataframe. we can save the analysis as a csv file
        # and load it with pandas after the init dialog
        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        # add menu bar
        self.menu_bar = QMenuBar(self)
        self.file_menu = self.menu_bar.addMenu("File")
        self.file_menu.addAction("Open Zarr Datastore...")
        self.file_menu.triggered.connect(self._show_init_dialog)
        self.setMenuBar(self.menu_bar)

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

        # left widgets -------------------------------------------------
        left_group = QGroupBox()
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(5)
        left_layout.addWidget(self.view)
        left_layout.addWidget(self._fov_table)

        # splitter for the plate map and the fov table
        self.splitter_top_left = QSplitter(self, orientation=Qt.Orientation.Vertical)
        self.splitter_top_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_top_left.setChildrenCollapsible(False)
        self.splitter_top_left.addWidget(self.view)
        self.splitter_top_left.addWidget(self._fov_table)
        top_left_group = QGroupBox()
        top_left_layout = QVBoxLayout(top_left_group)
        top_left_layout.setContentsMargins(10, 10, 10, 10)
        top_left_layout.addWidget(self.splitter_top_left)

        # splitter for the plate map/fov table and the image viewer
        self.splitter_bottom_left = QSplitter(self, orientation=Qt.Orientation.Vertical)
        self.splitter_bottom_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_bottom_left.setChildrenCollapsible(False)
        self.splitter_bottom_left.addWidget(top_left_group)
        self.splitter_bottom_left.addWidget(self._image_viewer)

        # right widgets --------------------------------------------------

        # tab widget
        self._tab = QTabWidget(self)
        self._tab.currentChanged.connect(self._on_tab_changed)

        # analysis tab
        self._analysis_tab = QWidget()
        self._tab.addTab(self._analysis_tab, "Analysis Tab")

        # plate map
        self._plate_map_dialog = QDialog(self)
        plate_map_layout = QHBoxLayout(self._plate_map_dialog)
        plate_map_layout.setContentsMargins(10, 10, 10, 10)
        plate_map_layout.setSpacing(5)
        self._plate_map_genotype = PlateMapWidget(self, title="Genotype Map")
        plate_map_layout.addWidget(self._plate_map_genotype)
        self._plate_map_treatment = PlateMapWidget(self, title="Treatment Map")
        plate_map_layout.addWidget(self._plate_map_treatment)

        self._plate_map_btn = QPushButton("Show/Edit Plate Map")
        self._plate_map_btn.setIcon(icon(MDI6.view_comfy))
        self._plate_map_btn.setIconSize(QSize(25, 25))
        self._plate_map_btn.clicked.connect(self._show_plate_map_dialog)
        self._plate_map_group = QGroupBox("Plate Map")
        plate_map_group_layout = QHBoxLayout(self._plate_map_group)
        plate_map_group_layout.setContentsMargins(10, 10, 10, 10)
        plate_map_group_layout.setSpacing(5)
        plate_map_group_layout.addWidget(self._plate_map_btn)
        plate_map_group_layout.addStretch(1)

        self._segmentation_wdg = _CellposeSegmentation(self)
        self._analysis_wdg = _AnalyseCalciumTraces(self)

        analysis_layout = QVBoxLayout(self._analysis_tab)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(15)
        analysis_layout.addWidget(self._plate_map_group)
        analysis_layout.addWidget(self._segmentation_wdg)
        analysis_layout.addWidget(self._analysis_wdg)
        analysis_layout.addStretch(1)

        # visualization tab
        self._visualization_tab = QWidget()
        self._tab.addTab(self._visualization_tab, "Single Wells Visualization Tab")
        visualization_layout = QGridLayout(self._visualization_tab)
        visualization_layout.setContentsMargins(5, 5, 5, 5)
        visualization_layout.setSpacing(5)

        self._graph_wdg_1 = _GraphWidget(self)
        self._graph_wdg_2 = _GraphWidget(self)
        self._graph_wdg_3 = _GraphWidget(self)
        self._graph_wdg_4 = _GraphWidget(self)
        self._graph_wdg_5 = _GraphWidget(self)
        self._graph_wdg_6 = _GraphWidget(self)
        visualization_layout.addWidget(self._graph_wdg_1, 0, 0)
        visualization_layout.addWidget(self._graph_wdg_2, 0, 1)
        visualization_layout.addWidget(self._graph_wdg_3, 0, 2)
        visualization_layout.addWidget(self._graph_wdg_4, 1, 0)
        visualization_layout.addWidget(self._graph_wdg_5, 1, 1)
        visualization_layout.addWidget(self._graph_wdg_6, 1, 2)

        self.GRAPHS = [
            self._graph_wdg_1,
            self._graph_wdg_2,
            self._graph_wdg_3,
            self._graph_wdg_4,
            self._graph_wdg_5,
            self._graph_wdg_6,
        ]

        # connect the roiSelected signal from the graphs to the image viewer so we can
        # highlight the roi in the image viewer when a roi is selected in the graph
        for graph in self.GRAPHS:
            graph.roiSelected.connect(self._highlight_roi)

        # splitter between the plate map/fov table/image viewer and the graphs
        self.main_splitter = QSplitter(self)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter_bottom_left)
        self.main_splitter.addWidget(self._tab)

        # add widgets to central widget
        self._central_widget_layout.addWidget(self.main_splitter)

        self.scene.selectedWellChanged.connect(self._on_scene_well_changed)

        self._loading_bar = _ProgressBarWidget(self, text="Loading Analysis Data...")

        self.showMaximized()

        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # data = "/Users/fdrgsp/Desktop/test/z.ome.zarr"
        # reader = OMEZarrReader(data)
        # data = "/Users/fdrgsp/Desktop/test/ts.tensorstore.zarr"
        # data = (
        #     r"/Volumes/T7 Shield/Neurons/NC240509_240523_Chronic/NC240509_240523_"
        #     "Chronic.tensorstore.zarr"
        # )
        # reader = TensorstoreZarrReader(data)
        # self._labels_path = "/Users/fdrgsp/Desktop/labels"
        # # # self._analysis_file_path = "/Users/fdrgsp/Desktop/analysis.json"
        # # self._analysis_file_path = "/Users/fdrgsp/Desktop/out"
        # self._analysis_file_path = "/Users/fdrgsp/Desktop/o1"
        # self._init_widget(reader)

    @property
    def datastore(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._datastore

    @datastore.setter
    def datastore(self, value: TensorstoreZarrReader | OMEZarrReader) -> None:
        self._datastore = value
        self._init_widget(value)

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, value: str | None) -> None:
        self._labels_path = value
        self._on_fov_table_selection_changed()

    @property
    def analysis_file_path(self) -> str | None:
        return self._analysis_file_path

    @analysis_file_path.setter
    def analysis_file_path(self, value: str) -> None:
        self._analysis_file_path = value
        self._load_analysis_data(value)

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        return self._analysis_data

    @analysis_data.setter
    def analysis_data(self, value: dict[str, dict[str, ROIData]]) -> None:
        self._analysis_data = value

    @property
    def labels(self) -> dict[str, np.ndarray]:
        return self._segmentation_wdg.labels

    def _show_plate_map_dialog(self) -> None:
        """Show the plate map dialog."""
        if self._plate_map_dialog.isHidden():
            self._plate_map_dialog.show()
        else:
            self._plate_map_dialog.raise_()
            self._plate_map_dialog.activateWindow()

    def _on_tab_changed(self, idx: int) -> None:
        """Update the grapg combo boxes when the tab is changed."""
        if idx != 1:
            return
        # get the current fov
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None:
            return
        # get the analysis data for the current fov if it exists
        analysis = self._analysis_data.get(str(value.fov.name), None)
        # update the graphs combo boxes
        self._update_graphs_combo(combo_red=(analysis is None))

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

    def _highlight_roi(self, roi: int) -> None:
        self._image_viewer._roi_number_le.setText(str(roi))
        self._image_viewer._highlight_rois()

    def _show_init_dialog(self) -> None:
        """Show a dialog to select a zarr datastore file and segmentation path."""
        init_dialog = _InitDialog(
            self,
            datastore_path=(
                str(self._datastore.path) if self._datastore is not None else None
            ),
            labels_path=self._labels_path,
            analysis_path=self._analysis_file_path,
        )
        if init_dialog.exec():
            datastore, self._labels_path, self._analysis_file_path = init_dialog.value()
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
                    f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                    f" {WRITERS[OME_ZARR][0]} are supported.",
                )
                return

            self._init_widget(reader)

    def _load_analysis_data(self, path: str | Path) -> None:
        """Load the analysis data from the given JSON file."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            show_error_dialog(
                self, f"Error while loading the file. Path {path} does not exist!"
            )
            return
        if not path.is_dir():
            show_error_dialog(
                self, f"Error while loading the file. Path {path} is not a directory!"
            )
            return

        # temporarily disable the whole widget
        self.setEnabled(False)

        # start the waiting progress bar
        self._loading_bar.setEnabled(True)
        self._loading_bar.setValue(0)
        self._loading_bar.show()

        create_worker(
            self._load_data_from_json,
            path=path,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress_bar,
                "finished": self._on_loading_finished,
                "errored": self._on_loading_finished,
            },
        )

    def _on_loading_finished(self) -> None:
        """Called when the loading of the analysis data is finished."""
        self._loading_bar.hide()
        # re-enable the whole widget
        self.setEnabled(True)

    # TODO: maybe use ThreadPoolExecutor
    def _load_data_from_json(self, path: Path) -> Generator[int, None, None]:
        """Load the analysis data from the given JSON file."""
        json_files = self._filter_data(list(path.glob("*.json")))
        self._loading_bar.setRange(0, len(json_files))
        try:
            # loop over the files in the directory
            for idx, f in enumerate(tqdm(json_files, desc="Loading Analysis Data")):
                yield idx + 1
                # get the name of the file without the extensions
                well = f.name.removesuffix(f.suffix)
                # create the dict for the well
                self._analysis_data[well] = {}
                # open the data for the well
                with open(f) as file:
                    try:
                        data = cast(dict, json.load(file))
                    except json.JSONDecodeError as e:
                        show_error_dialog(self, f"Error loading the analysis data: {e}")
                        self._analysis_data = data
                    # if the data is empty, continue
                    if not data:
                        continue
                    # loop over the rois
                    for roi in data.keys():
                        # get the data for the roi
                        roi_data = cast(dict, data[roi])
                        # if there are peaks, convert them to Peaks objects
                        if peaks := roi_data.get("peaks", {}):
                            peaks_objects = []
                            for p in peaks:
                                peaks_objects.append(Peaks(**p))
                                roi_data["peaks"] = peaks_objects
                        # convert to a ROIData object and add store it in _analysis_data
                        self._analysis_data[well][roi] = ROIData(**roi_data)
        except Exception as e:
            show_error_dialog(self, f"Error loading the analysis data: {e}")
            self._analysis_data.clear()

    def _filter_data(self, path_list: list[Path]) -> list[Path]:
        # the json file names should be in the form A1_0000.json
        for f in path_list:
            name = f.name.removesuffix(f.suffix)  # A1_0000
            if name in {GENOTYPE_MAP, TREATMENT_MAP}:
                path_list.remove(f)
                continue
            split_name = name.split("_")  # ["A1", "0000"]
            if len(split_name) != 2:
                path_list.remove(f)
                continue
            well, pos = split_name
            if not re.match(r"^[a-zA-Z0-9]+$", well):  # only letters and numbers
                path_list.remove(f)
                continue
            if not pos.isdigit():  # only digits
                path_list.remove(f)
                continue
        return path_list

    def _update_progress_bar(self, value: int) -> None:
        """Update the progress bar value."""
        self._loading_bar.setValue(value)

    def _init_widget(self, reader: TensorstoreZarrReader | OMEZarrReader) -> None:
        """Initialize the widget with the given datastore."""
        # load analysis json file if the path is not None
        if self._analysis_file_path:
            self._load_analysis_data(self._analysis_file_path)

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

        # set the segmentation widget data
        self._segmentation_wdg.data = self._datastore
        self._segmentation_wdg._output_path._path.setText(self._labels_path)
        # set the analysis widget data
        self._analysis_wdg.data = self._datastore
        self._analysis_wdg.labels_path = self._labels_path
        self._analysis_wdg._output_path._path.setText(self._analysis_file_path)

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

        self._load_plate_map(plate)

    def _load_plate_map(self, plate: Plate) -> None:
        """Load the plate map from the given file."""
        self._plate_map_genotype.clear()
        self._plate_map_treatment.clear()
        self._plate_map_genotype.setPlate(plate)
        self._plate_map_treatment.setPlate(plate)
        # load plate map if exists
        if self._analysis_file_path is not None:
            gen_path = Path(self._analysis_file_path) / GENOTYPE_MAP
            if gen_path.exists():
                self._plate_map_genotype.setValue(gen_path)
            treat_path = Path(self._analysis_file_path) / TREATMENT_MAP
            if treat_path.exists():
                self._plate_map_treatment.setValue(treat_path)

    def _on_scene_well_changed(self, value: Well | None) -> None:
        """Update the FOV table when a well is selected."""
        self._fov_table.clear()
        self._image_viewer._clear_highlight()

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
        self._image_viewer._clear_highlight()
        value = self._fov_table.value() if self._fov_table.selectedItems() else None

        if value is None:
            self._image_viewer.setData(None, None)
            self._update_graphs_combo(combo_red=True, clear=True)
            return

        if self._datastore is None:
            return

        if not self._datastore.sequence:
            return

        # get a single frame for the selected FOV (at 2/3 of the time points)
        t = int(len(self._datastore.sequence.stage_positions) / 3 * 2)
        data = cast(np.ndarray, self._datastore.isel(p=value.pos_idx, t=t, c=0))

        # get one random segmentation between 0 and 2
        labels = self._get_labels(value)
        analysis = self._analysis_data.get(str(value.fov.name), None)
        # flip data and labels vertically or will look different from the StackViewer
        data = np.flip(data, axis=0)
        labels = np.flip(labels, axis=0) if labels is not None else None
        self._image_viewer.setData(data, labels)
        self._set_graphs_fov(value)

        self._update_graphs_combo(
            combo_red=(analysis is None), clear=(analysis is None)
        )

    def _set_graphs_fov(self, value: WellInfo | None) -> None:
        """Set the FOV title for the graphs."""
        if value is None:
            return
        title = value.fov.name or f"Position {value.pos_idx}"
        self._update_graphs_combo(set_title=title)

    def _get_labels(self, value: WellInfo) -> np.ndarray | None:
        """Get the labels for the given FOV."""
        if self._labels_path is None:
            return None
        # the labels tif file should have the same name as the position
        # and should end with _on where n is the position number (e.g. C3_0000_p0.tif)
        pos_idx = f"p{value.pos_idx}"
        pos_name = value.fov.name
        for f in Path(self._labels_path).iterdir():
            name = f.name.replace(f.suffix, "")
            if pos_name and pos_name in f.name and name.endswith(f"_{pos_idx}"):
                return tifffile.imread(f)  # type: ignore
        return None

    def _on_fov_double_click(self) -> None:
        """Open the selected FOV in a new StackViewer window."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None or self._datastore is None:
            return

        data = self._datastore.isel(p=value.pos_idx)
        viewer = StackViewer(data, parent=self)
        viewer.setWindowTitle(value.fov.name or f"Position {value.pos_idx}")
        viewer.setWindowFlag(Qt.WindowType.Dialog)
        viewer.show()

    def _update_graphs_combo(
        self,
        set_title: str | None = None,
        combo_red: bool = False,
        clear: bool = False,
    ) -> None:
        for graph in self.GRAPHS:
            if set_title is not None:
                graph.fov = set_title

            if clear:
                graph.clear_plot()

            graph.set_combo_text_red(combo_red)
