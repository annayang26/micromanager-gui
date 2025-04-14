from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
import useq
import xlsxwriter
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._plate_viewer._init_dialog import _BrowseWidget
from micromanager_gui._widgets._mda_widget._save_widget import (
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
)

from ._plate_viewer._logger import LOGGER
from ._plate_viewer._plate_map import PlateMapData
from ._plate_viewer._util import (
    COND1,
    COND2,
    GENOTYPE_MAP,
    GREEN,
    RED,
    STIMULATION_MASK,
    TREATMENT_MAP,
    ROIData,
    calculate_dff,
    create_stimulation_mask,
    get_iei,
    get_linear_phase,
    get_overlap_roi_with_stimulated_area,
    show_error_dialog,
)
from .readers import OMEZarrReader, TensorstoreZarrReader

if TYPE_CHECKING:
    from collections.abc import Generator
    from threading import Event

    from superqt.utils import GeneratorWorker

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
DATA_KEY = 0
LABELS = ("_labels", "_label")
EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])
CAMERA_KEY = "camera_metadata"
ELAPSED_TIME_KEY = "ElapsedTime-ms"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
MIN_PEAKS_HEIGHT = 0.0
STIMULATION_AREA_THRESHOLD = 0.5  # 50%

class BatchAnalysis(QWidget):
    def __init__(
            self,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)

            self.setWindowTitle("Batch Process")
            self._stop_event: Event = Manager().Event()

            self._plate_map_data: dict[str, dict[str, str]] = {}
            self._stimulated_area_mask: np.ndarray | None = None
            self._genotype_pm: list[PlateMapData] | None = None
            self._treatment_pm: list[PlateMapData] | None = None

            self.recording_folder_path: list[str] = []
            self.labels_folder_path: list[str] = []
            self._analysis_data: dict = {}

            self._run_worker: GeneratorWorker | None = None

            self._input_path = _BrowseWidget(
                self,
                "Input Folder",
                "",
                "Choose the folder containing the files to analyze.",
                is_dir=True,
            )

            self._genotype_pm_path = _BrowseWidget(
                self,
                "Genotype PlateMap",
                "",
                "Make sure to choose the genotype platemap for the entire experiment!",
                is_dir=False,
            )

            self._treatment_pm_path = _BrowseWidget(
                self,
                "Treatment PlateMap",
                "",
                "Make sure to choose the treatment platemap for the entire experiment!",
                is_dir=False,
            )

            experiment_type_wdg = QWidget(self)
            experiment_type_wdg_layout = QHBoxLayout(experiment_type_wdg)
            experiment_type_wdg_layout.setContentsMargins(0, 0, 0, 0)
            experiment_type_wdg_layout.setSpacing(5)
            activity_combo_label = QLabel("Experiment Type:")
            activity_combo_label.setSizePolicy(*FIXED)
            self._experiment_type_combo = QComboBox()
            self._experiment_type_combo.addItems([SPONTANEOUS, EVOKED])
            self._experiment_type_combo.currentTextChanged.connect(
                self._on_activity_changed
            )
            experiment_type_wdg_layout.addWidget(activity_combo_label)
            experiment_type_wdg_layout.addWidget(self._experiment_type_combo)

            self._stimulation_area_path = _BrowseWidget(
                self,
                label="Stimulated Area File",
                tooltip=(
                    "Select the path to the image of the stimulated area.\n"
                    "The image should either be a binary mask or a grayscale image where "
                    "the stimulated area is brighter than the rest.\n"
                    "Accepted formats: .tif, .tiff."
                ),
                is_dir=False,
            )
            self._stimulation_area_path.hide()

            buttons_wdg = QWidget(self)
            buttons_wdg.setSizePolicy(*FIXED)
            buttons_layout = QHBoxLayout(buttons_wdg)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(5)
            self._run_btn = QPushButton("Run")
            self._run_btn.setSizePolicy(*FIXED)
            self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
            self._run_btn.setIconSize(QSize(25, 25))
            self._run_btn.clicked.connect(self._run)
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.setSizePolicy(*FIXED)
            self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))
            self._cancel_btn.setIconSize(QSize(25, 25))
            self._cancel_btn.clicked.connect(self.cancel)
            buttons_layout.addStretch(1)
            buttons_layout.addWidget(self._run_btn)
            buttons_layout.addWidget(self._cancel_btn)

            self.groupbox = QGroupBox("Batch Analysis", self)
            settings_groupbox_layout = QGridLayout(self.groupbox)
            settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
            settings_groupbox_layout.setSpacing(5)
            settings_groupbox_layout.addWidget(self._input_path, 0, 0, 1, 2)
            settings_groupbox_layout.addWidget(self._genotype_pm_path, 1, 0, 1, 2)
            settings_groupbox_layout.addWidget(self._treatment_pm_path, 2, 0, 1, 2)
            settings_groupbox_layout.addWidget(experiment_type_wdg, 3, 0, 2, 1)
            settings_groupbox_layout.addWidget(self._stimulation_area_path, 5, 0, 2, 1)
            settings_groupbox_layout.addWidget(buttons_wdg, 7, 0, 2, 1)
            # settings_groupbox_layout.addWidget(buttons_wdg, 3, 0, 2, 1)

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(self.groupbox)
            main_layout.addStretch(1)

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulated area path widget."""
        (
            self._stimulation_area_path.show()
            if text == EVOKED
            else self._stimulation_area_path.hide()
        )

    def cancel(self) -> None:
        """Cancel the current run."""
        self._stop_event.set()
        if self._run_worker is not None:
            self._run_worker.quit()
            self._plate_map_data = {}
            self._genotype_pm = None
            self._treatment_pm = None

    def run(self) -> None:
        self._stop_event.clear()
        self._run_worker = create_worker(self._run, _start_thread=True)

    def _run(self) -> None:
        """Run the batch analysis."""
        input_path = self._input_path.value()
        # input_path = r'/Volumes/Expansion/test'

        # check stimulate activity
        if self._is_stimulated() and (
            not self._prepare_stimulation_mask(self._stimulation_area_path)):
            return None

        stimulated = self._is_stimulated()

        if not input_path:
            return

        self._load_plate_map()
        self._handle_plate_map()

        for folder in Path(input_path).iterdir():
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.name.endswith(EXT):
                        self.recording_folder_path.append(str(f))
                    if f.name.endswith(LABELS):
                        self.labels_folder_path.append(str(f))

                # TODO: uncomment the following line when running actual analysis
                # if len(self.recording_folder_path) != len(self.labels_folder_path):
                #     print(
                #         f"{self.recording_folder_path[-1]} is not segmented. Please run segmentaiton first."  # noqa: E501
                #         )
                #     self.cancel()
                #     break

        print("Number of folders to analyze: %s", len(self.recording_folder_path))

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)

        for f, label in zip(self.recording_folder_path,
                        self.labels_folder_path):
            self._analysis_folder(f, label, stimulated)

        # try:
        #     with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        #         futures = [
        #             executor.submit(
        #                 self._analysis_folder,
        #                 data_folder=f,
        #                 label_folder=label,
        #                 stimulated=stimulated)
        #             for f, label in zip(self.recording_folder_path,
        #                 self.labels_folder_path)
        #         ]
        #         print(f"length of future: {len(futures)}")

        #         for future in tqdm(as_completed(futures),
        #                 total=len(self.recording_folder_path),
        #                 desc="Processing files",
        #             ):
        #             if self._check_for_abort_requested():
        #                 print("Abort requested, cancelling all futures...")
        #                 for f in futures:
        #                     f.cancel()
        #                 break
        #             try:
        #                 future.result()
        #             except Exception as e:
        #                 print(f'future result: {future.result()} at {f}')
        #                 print(f"An error occurred inside: {e}")
        #                 break
        #         print("All tasks completed.")

        # except Exception as e:
        #     print("An error occurred: %s", e)

        # for f, label in zip(self.recording_folder_path, self.labels_folder_path):
            # self._analysis_data = {}
            # print(f"start with {f}")
            # self._run_worker = create_worker(
            #     self._analysis_folder,
            #     data_folder=f,
            #     label_folder=label,
            #     stimulated=stimulated,
            #     _start_thread=True,
            #     _connect={
            #         # "yielded": show_error_dialog,
            #         "finished": self._on_worker_finished,
            #         "errored": self._on_worker_finished,
            #     },
            #     )
            # print(f"{f} ended")

    def _check_for_abort_requested(self) -> bool:
        return bool(self._run_worker is not None and self._run_worker.abort_requested)

    def _load_plate_map(self) -> None:
        """Load plate map from the given path."""
        if self._genotype_pm_path is not None:
            self._genotype_pm = self._setValue(self._genotype_pm_path.value())
        if self._treatment_pm_path is not None:
            self._treatment_pm = self._setValue(self._treatment_pm_path.value())

    def _handle_plate_map(self) -> None:
        """Handle plate map data."""
        if self._genotype_pm and self._treatment_pm:
            for data in self._genotype_pm:
                self._plate_map_data[data.name] = {COND1: data.condition[0]}

            for data in self._treatment_pm:
                if data.name in self._plate_map_data:
                    self._plate_map_data[data.name][COND2] = data.condition[0]
                else:
                    self._plate_map_data[data.name] = {COND2: data.condition[0]}

    def _setValue(self, path: Path | str) -> list[PlateMapData]:
        """Set values in the plate map."""
        if isinstance(path, (Path, str)):
            with open(path) as pmap:
                data = json.load(pmap)
            data_list = cast(list, data)

        pm_data_list = []
        for data in data_list:
            # convert the data to a PlateMapData object if it is a list of strings
            if not isinstance(data, PlateMapData):
                data = PlateMapData(*data)
            pm_data_list.append(data)

        return pm_data_list

    def _is_stimulated(self) -> bool:
        """Return True if the activity type is evoked."""
        activity_type = self._experiment_type_combo.currentText()
        return activity_type == EVOKED  # type: ignore

    def _prepare_stimulation_mask(self, analysis_path: Path) -> bool:
        """Generate the stimulation mask if the experiment involves evoked activity."""
        if stim_area_file := self._stimulation_area_path.value():
            self._stimulated_area_mask = create_stimulation_mask(stim_area_file)
            stim_mask_path = analysis_path / STIMULATION_MASK
            tifffile.imwrite(str(stim_mask_path), self._stimulated_area_mask)
            return True

        self._stimulated_area_mask = None
        show_error_dialog("No Stimulated Area File Provided!")
        return False

    def _on_worker_finished(self) -> None:
        """Called when the extraction is finished."""
        print("Extraction of traces finished.")

        self._enable(True)

        # # show a message box if there are failed labels
        # if self._failed_labels:
        #     msg = (
        #         "The following labels were not found during the analysis:\n\n"
        #         + "\n".join(self._failed_labels)
        #     )
        #     self._show_and_log_error(msg)

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._input_path.setEnabled(True)
        self._genotype_pm_path.setEnabled(enable)
        self._stimulation_area_path.setEnabled(enable)
        self._experiment_type_combo.setEnabled(enable)
        self._treatment_pm_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)

    def _analysis_folder(self,
                        data_folder: str,
                        label_folder: str,
                        stimulated: bool=False,
                        ) -> Generator[str, None, None]:
        """Analyze data."""
        if self._stop_event.is_set():
            print(f"Analysis process stopped for {data_folder}")
            return
        print("Start analyzing ", data_folder)

        # get the recording data
        path = Path(data_folder)
        data: OMEZarrReader | TensorstoreZarrReader
        if path.name.endswith(WRITERS[OME_ZARR][0]):
            data = OMEZarrReader(path)
        elif path.name.endswith(WRITERS[ZARR_TESNSORSTORE][0]):
            data = TensorstoreZarrReader(path)
        else:
            print(f"Unsupported file format: {path.name}, skipping...")
            return

        # get sequence
        sequence = data.sequence
        if sequence is None:
            print(f"Skipping {data.path.name}, no sequence foundata.")
            return

        # divide the files in the folder to smaller chuncks
        positions = self._get_positions_to_analyze(label_folder, data)
        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system
        pos = len(positions)
        chunk_size = max(1, pos // cpu_count)

        print("CPU count: ", cpu_count)
        print("Chunk size: ", chunk_size)

        file_name = data.path.name
        for ext in EXT:
            if file_name.endswith(ext):
                file_name = file_name[: -len(ext)]
                break

        path = data.path.parent / f"{file_name}_output"
        if not path.exists():
            path.mkdir()

        self._save_plate_maps(path=path)

        self._analysis_data = {}

        # self._run_worker = create_worker(
        #         self._analysis_file_for_chunk,
        #         data,
        #         label_folder,
        #         path,
        #         positions,
        #         stimulated,
        #         _start_thread=True,
        #         _connect={
        #             # "yielded": show_error_dialog,
        #             "finished": self._on_worker_finished,
        #             "errored": self._on_worker_finished,
        #         },
        #     )

        try:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(
                        self._analysis_file_for_chunk,
                        data,
                        label_folder,
                        path,
                        positions,
                        stimulated,
                        start,
                        min(start + chunk_size, pos),
                        )
                        for start in range(0, pos, chunk_size)
                    ]

                for idx, future in enumerate(as_completed(futures)):
                    if self._check_for_abort_requested():
                        print("Abort requested, cancelling all futures...")
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                        print(f"Chunk {idx + 1} completed.")
                    except Exception as e:
                        print(f'future result: {future.result()} at {f}')
                        print(f"An error occurred inside: {e}")
                        break

            print("All tasks completed.")

        except Exception as e:
            print("An error occurred:", e)

        # try:
        #     output_csv(output_path=path,
        #             analysis_data=analysis_data,
        #             pm_data=pm_data,
        #             )
        # except Exception as e:
        #     print("CSV files failed to compile! Error: %s", e)

    def _save_plate_maps(self,
                        path: str) -> None:
        geno_path = Path(path) / GENOTYPE_MAP
        cond_path = Path(path) / TREATMENT_MAP
        with geno_path.open("w") as f1:
            json.dump(self._genotype_pm, f1, indent=2)
        with cond_path.open("w") as f2:
            json.dump(self._treatment_pm, f2, indent=2)

    def _get_labels_file(self, label_folder: str, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        if label_folder is None:
            return None
        for label_file in Path(label_folder).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    def _get_positions_to_analyze(self,
                                  label_folder: str,
                                  data: OMEZarrReader | TensorstoreZarrReader
                                ) -> list[int] | None:
        """Get the positions to analyze."""
        if data is None or (sequence := data.sequence) is None:
            return None

        positions = [
            i
            for i, p in enumerate(sequence.stage_positions)
            if self._get_labels_file(
                label_folder,
                f"{p.name or f'pos_{str(i).zfill(4)}'}_p{i}.tif"
            )
        ]

        return positions

    def _analysis_file_for_chunk(
        self,
        data: OMEZarrReader | TensorstoreZarrReader,
        label_folder: str,
        output_path: str,
        positions: list[int],
        stimulated: bool,
        start: int,
        end: int
    ) -> None:
        """Extract the roi traces for the given chunk."""
        print("inside analysis chunk")
        for p in range(start, end):
            if self._check_for_abort_requested():
                break
            self._analysis_trace_data_per_position(
                data,
                label_folder,
                output_path,
                positions[p],
                stimulated)

    def _analysis_trace_data_per_position(
            self,
            data_series: OMEZarrReader | TensorstoreZarrReader,
            label_folder: str,
            output_path: Path,
            p: int,
            stimulated: bool = False,
        ) -> dict[str, dict[str, ROIData]]:
        """Extract the roi traces for the given position."""
        if data_series is None or self._check_for_abort_requested():
            return

        # get the data and metadata for the position
        data, meta = data_series.isel(p=p, metadata=True)

        # the "Event" key was used in the old metadata format
        event_key = "mda_event" if "mda_event" in meta[0] else "Event"

        # get the fov_name name from metadata
        fov_name = self._get_fov_name(event_key, meta, p)

        # create the dict for the fov if it does not exist
        if fov_name not in self._analysis_data:
            self._analysis_data[fov_name] = {}
        # get the labels file for the position
        labels_path = self._get_labels_file_for_position(label_folder, fov_name, p)
        if labels_path is None:
            return

        # open the labels file and create masks for each label
        labels = tifffile.imread(labels_path)
        labels_masks = self._create_label_masks_dict(labels)
        sequence = cast(useq.MDASequence, data_series.sequence)

        # get the elapsed time from the metadata to calculate the total time in seconds
        elapsed_time_list = self.get_elapsed_time_list(meta)

        # get the exposure time from the metadata
        exp_time = meta[0][event_key].get("exposure", 0.0)

        # get timepoints
        timepoints = sequence.sizes["t"]

        # get the total time in seconds for the recording
        tot_time_sec = self._calculate_total_time(
            elapsed_time_list, exp_time, timepoints
        )
        print(f"total time: {tot_time_sec}")
        print(f"Extracting Traces from Well {fov_name}.")
        for label_value, label_mask in tqdm(
            labels_masks.items(), desc=f"Extracting Traces from Well {fov_name}"
        ):
            if self._check_for_abort_requested():
                break

            # extract the data
            self._analysis_data[fov_name] = self._process_roi_trace(
                data,
                meta,
                fov_name,
                label_value,
                label_mask,
                timepoints,
                exp_time,
                tot_time_sec,
                stimulated,
                elapsed_time_list,
            )

        # save the analysis data for the well
        self._save_analysis_data(self._analysis_data, output_path, fov_name)

    def _process_roi_trace(
            self,
            data: np.ndarray,
            meta: list[dict],
            fov_name: str,
            label_value: int,
            label_mask: np.ndarray,
            timepoints: int,
            exp_time: float,
            tot_time_sec: float,
            stimulated: bool,
            elapsed_time_list: list[float],
        ) -> None:
        """Process individual ROI traces."""
        # calculate the mean trace for the roi
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        px_size = meta[0].get("PixelSizeUm", None)
        # calculate the size of the roi in µm if px_size is available or not 0,
        # otherwise use the size is in pixels
        roi_size = roi_size_pixel * px_size if px_size else roi_size_pixel

        # exclude small rois, might not be necessary if trained cellpose performs
        # better
        if px_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
            return

        # check if the roi is stimulated
        roi_stimulation_overlap_ratio = 0.0
        if stimulated and self._stimulated_area_mask is not None:
            roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                self._stimulated_area_mask, label_mask
            )

        # compute the mean for each frame
        roi_trace: np.ndarray = masked_data.mean(axis=1)

        # calculate the dff of the roi trace
        dff: np.ndarray = calculate_dff(roi_trace, window=10, plot=False)

        # deconvolve the dff trace
        dec_dff, spikes, _, _, _ = deconvolve(dff, penalty=1)

        noise_level_dec_dff = np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
        peaks_prominence_dec_dff = noise_level_dec_dff  # * 2

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff, prominence=peaks_prominence_dec_dff, height=MIN_PEAKS_HEIGHT
        )

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [dec_dff[p] for p in peaks_dec_dff]

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = len(peaks_dec_dff) / tot_time_sec if tot_time_sec else None

        # get the conditions for the well
        condition_1, condition_2 = self._get_conditions(fov_name)

        # get the linear and cubic phase of the peaks in the dec_dff trace
        linear_phase = get_linear_phase(timepoints, peaks_dec_dff)

        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # store the data to the analysis dict as ROIData
        self._analysis_data[fov_name][str(label_value)] = ROIData(
            well_fov_position=fov_name,
            raw_trace=roi_trace.tolist(),  # type: ignore
            dff=dff.tolist(),  # type: ignore
            dec_dff=dec_dff.tolist(),
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            dec_dff_frequency=frequency or None,
            inferred_spikes=spikes.tolist(),
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
            condition_1=condition_1,
            condition_2=condition_2,
            total_recording_time_in_sec=tot_time_sec,
            active=len(peaks_dec_dff) > 0,
            linear_phase=linear_phase,
            iei=iei,
            stimulated=roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD,
        )

    def _get_fov_name(self, event_key: str, meta: list[dict], p: int) -> str:
        """Retrieve the fov name from metadata."""
        # the "Event" key was used in the old metadata format
        pos_name = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
        return f"{pos_name}_p{p}"

    def _get_labels_file_for_position(self,
                                      label_folder: str, fov: str, p: int
                                      ) -> str | None:
        """Retrieve the labels file for the given position."""
        # if the fov name does not end with "_p{p}", add it
        _failed_labels = []
        labels_name = f"{fov}.tif" if fov.endswith(f"_p{p}") else f"{fov}_p{p}.tif"
        labels_path = self._get_labels_file(label_folder, labels_name)
        if labels_path is None:
            _failed_labels.append(labels_name)
            print(f"No labels found for {labels_name}!")
        return labels_path, _failed_labels

    def _create_label_masks_dict(self, labels: np.ndarray) -> dict:
        """Create masks for each label in the labels image."""
        # get the range of labels and remove the background (0)
        labels_range = np.unique(labels[labels != 0])
        return {label_value: (labels == label_value) for label_value in labels_range}

    def get_elapsed_time_list(self, meta: list[dict]) -> list[float]:
        """Get the elapsed time for each timepoint to calculate tot_time_sec."""
        elapsed_time_list: list[float] = []
        if (cam_key := CAMERA_KEY) in meta[0]:  # new metadata format
            for m in meta:
                et = m[cam_key].get(ELAPSED_TIME_KEY)
                if et is not None:
                    elapsed_time_list.append(float(et))
        else:  # old metadata format
            for m in meta:
                et = m.get(ELAPSED_TIME_KEY)
                if et is not None:
                    elapsed_time_list.append(float(et))
        return elapsed_time_list

    def _calculate_total_time(
        self,
        elapsed_time_list: list[float],
        exp_time: float,
        timepoints: int,
    ) -> float:
        """Calculate total time in seconds for the recording."""
        # if the len of elapsed time is not equal to the number of timepoints,
        # use exposure time and the number of timepoints to calculate tot_time_sec
        if len(elapsed_time_list) != timepoints:
            tot_time_sec = exp_time * timepoints / 1000
        # otherwise, calculate the total time in seconds using the elapsed time.
        # NOTE: adding the exposure time to consider the first frame
        else:
            tot_time_sec = (
                elapsed_time_list[-1] - elapsed_time_list[0] + exp_time
            ) / 1000
        return tot_time_sec

    def _output_csv():
        """Compile data and output into CSV files."""

    def _get_conditions(self, pos_name: str) -> tuple[str | None, str | None]:
        """Get the conditions for the well if any."""
        condition_1 = condition_2 = None
        if self._plate_map_data:
            well_name = pos_name.split("_")[0]
            if well_name in self._plate_map_data:
                condition_1 = self._plate_map_data[well_name].get(COND1)
                condition_2 = self._plate_map_data[well_name].get(COND2)
            else:
                condition_1 = condition_2 = None
        return condition_1, condition_2

    def _save_analysis_data(self, analysis_data: dict,
                            output_path: str, pos_name: str) -> None:
        """Save analysis data to a JSON file."""
        print("Saving JSON file for Well %s.", pos_name)
        path = Path(output_path) / f"{pos_name}.json"
        with path.open("w") as f:
            json.dump(
                analysis_data[pos_name],
                f,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )