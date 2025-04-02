from __future__ import annotations

import concurrent.futures
import json
import os
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
    QLabel,
    QGridLayout,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._plate_viewer._init_dialog import _BrowseWidget
from micromanager_gui._widgets._mda_widget._save_widget import (
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
)

from ._plate_viewer._plate_map import PlateMapData
from ._plate_viewer._util import (
    COND1,
    COND2,
    GENOTYPE_MAP,
    GREEN,
    RED,
    TREATMENT_MAP,
    ROIData,
)
from .readers import OMEZarrReader, TensorstoreZarrReader

if TYPE_CHECKING:
    from threading import Event

    from superqt.utils import FunctionWorker

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
DATA_KEY = 0
LABELS = "_labels"
EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])
CAMERA_KEY = "camera_metadata"
ELAPSED_TIME_KEY = "ElapsedTime-ms"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"

class BatchAnalysis(QWidget):
    def __init__(
            self,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)

            self.setWindowTitle("Batch Process")

            self._run_worker: FunctionWorker | None = None
            self._futures: list[concurrent.futures.Future] = []
            self._stop_event: Event = Manager().Event()

            self._plate_map_data: dict[str, dict[str, str]] = {}
            self._genotype_pm: list[PlateMapData] | None = None
            self._treatment_pm: list[PlateMapData] | None = None


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
                is_dir=False
            )

            self._treatment_pm_path = _BrowseWidget(
                self,
                "Treatment PlateMap",
                "",
                "Make sure to choose the treatment platemap for the entire experiment!",
                is_dir=False
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
            self._run_btn.clicked.connect(self.run)
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

    def run(self) -> None:
        """Run the worker."""
        self._stop_event.clear()
        print("run")
        # self._run_worker = create_worker(self._run, _start_thread=True)

    def cancel(self) -> None:
        """Cancel the current run."""
        self._stop_event.set()
        for future in self._futures:
            future.cancel()
        if self._run_worker is not None:
            self._run_worker.quit()
            self._plate_map_data = {}
            self._genotype_pm = None
            self._treatment_pm = None

    def _run(self) -> None:
        """Run the batch analysis."""
        input_path = self._input_path.value()
        # input_path = r'/Volumes/Expansion/test'

        if not input_path:
            return

        self._load_plate_map()
        self._handle_plate_map()

        recording_folder_path = []
        labels_folder_path = []
        for folder in Path(input_path).iterdir():
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.name.endswith(EXT):
                        recording_folder_path.append(str(f))
                    if f.name.endswith(LABELS):
                        labels_folder_path.append(str(f))

                # TODO: uncomment the following line when running actual analysis
                if len(recording_folder_path) != len(labels_folder_path):
                    print(
                        f"{recording_folder_path[-1]} is not segmented. Please run segmentaiton first."  # noqa: E501
                        )
                    self.cancel()
                    break

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpu_count
                ) as executor:
                for f, label in zip(recording_folder_path, labels_folder_path):
                    # futures = [
                    #     executor.submit(_analyze_data, f, label,
                    futures = executor.submit(_analyze_data, f, label,
                                            self._plate_map_data, self._genotype_pm,
                                            self._treatment_pm, self._stop_event)
                    self._futures.append(futures)

                for future in tqdm(
                    concurrent.futures.as_completed(self._futures),
                    total=len(recording_folder_path),
                    desc="Processing files",
                ):
                    try:
                        future.result()
                    except Exception as e:
                        print(f'future result: {future.result()} at {f}')
                        print(f"An error occurred inside: {e}")

        except Exception as e:
            print("An error occurred: %s", e)

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
                self._plate_map_data[data.name] = {COND1: data.condition}

            for data in self._treatment_pm:
                if data.name in self._plate_map_data:
                    self._plate_map_data[data.name][COND2] = data.condition
                else:
                    self._plate_map_data[data.name] = {COND2: data.condition}

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

def _analyze_data(data_folder: str, label_folder: str,
                  pm_data: dict[str, dict[str, str]],
                  geno_pm: list[PlateMapData],
                  cond_pm: list[PlateMapData],
                  stop_event: Event) -> None:
    """Analyze data."""
    if stop_event.is_set():
        print(f"Analysis process stopped for {data_folder}")
        return

    path = Path(data_folder)
    data: OMEZarrReader | TensorstoreZarrReader
    if path.name.endswith(WRITERS[OME_ZARR][0]):
        data = OMEZarrReader(path)
    elif path.name.endswith(WRITERS[ZARR_TESNSORSTORE][0]):
        data = TensorstoreZarrReader(path)
    else:
        print(f"Unsupported file format: {path.name}, skipping...")
        return

    sequence = data.sequence
    if sequence is None:
        print(f"Skipping {data.path.name}, no sequence foundata.")
        return

    positions = list(range(len(sequence.stage_positions)))

    file_name = data.path.name
    for ext in EXT:
        if file_name.endswith(ext):
            file_name = file_name[: -len(ext)]
            break

    path = data.path.parent / f"{file_name}_output"
    if not path.exists():
        path.mkdir()

    _save_plate_maps(geno_pm=geno_pm,
                    cond_pm=cond_pm,
                    path=path)

    analysis_data = _analyze(
        data=data,
        labels_path=label_folder,
        pm_data=pm_data,
        output_path=path,
        positions=positions,
        stop_event=stop_event
    )
    # try:
    #     output_csv(output_path=path,
    #             analysis_data=analysis_data,
    #             pm_data=pm_data,
    #             )
    # except Exception as e:
    #     print("CSV files failed to compile! Error: %s", e)

def _save_plate_maps(geno_pm: list[PlateMapData],
                    cond_pm: list[PlateMapData],
                    path: str) -> None:
    geno_path = Path(path) / GENOTYPE_MAP
    cond_path = Path(path) / TREATMENT_MAP
    with geno_path.open("w") as f1:
        json.dump(geno_pm, f1, indent=2)
    with cond_path.open("w") as f2:
        json.dump(cond_pm, f2, indent=2)

def _analyze(
    data: OMEZarrReader | TensorstoreZarrReader,
    labels_path: str,
    pm_data: dict[str, dict[str, str]],
    output_path: Path,
    positions: list[int],
    stop_event: Event,
) -> dict[str, dict[str, ROIData]]:
    analysis_data: dict[str, dict[str, ROIData]] = {}
    for p in tqdm(positions, desc="Processing positions"):
        if stop_event.is_set():
            print(f"Analysis stopped at position {p}")
            break

        # get the data
        stack, meta = data.isel(p=p, metadata=True)
        if stack is None or meta is None:
            print("No data found for %s!", p)
            continue

        # the "Event" key was used in the old metadata format
        event_key = "mda_event" if "mda_event" in meta[0] else "Event"

        # get the fov_name name from metadata
        pos_name = _get_fov_name(event_key, meta, p)

        if pos_name not in analysis_data:
            analysis_data[pos_name] = {}

        # total_frames = stack.shape[0]
        # binning, magnification, pixel_size, objective,\
        #     exposure, framerate = _extract_metadata(meta) # exposure time in ms
        # framerate *= 1000 # seconds
        # recording_time = total_frames/framerate # in seconds

        # matching label name
        labels_path, failed_labels = _get_labels_file_for_position(pos_name, p)
        if labels_path is None:
            return

        # open the labels file and create masks for each label
        labels = tifffile.imread(labels_path)
        labels_masks = _create_label_masks_dict(labels)
        sequence = cast(useq.MDASequence, data.sequence)

        # get the elapsed time from the metadata to calculate the total time in seconds
        elapsed_time_list = get_elapsed_time_list(meta)

        # get the exposure time from the metadata
        exp_time = meta[0][event_key].get("exposure", 0.0)

        # get timepoints
        timepoints = sequence.sizes["t"]

        # get the total time in seconds for the recording
        tot_time_sec = _calculate_total_time(
            elapsed_time_list, exp_time, timepoints
        )

        # check if it is an evoked activity experiment
        stimulated = _is_stimulated()

        # temporary storage for trace to use for photobleaching correction
        # fitted_curves: list[tuple[list[float], list[float], float]] = []

        roi_trace: np.ndarray | list[float] | None
        roi_size_um: float | None
        small_rois: list[int] = []

        average_trace = cast(np.ndarray, stack.mean(axis=(1, 2)))
        avg_exponential_decay = _get_exponential_decay(average_trace)

        # temporary storage for trace to use for photobleaching correction
        top_exponential_decay = (None, None, 0)

        # extract roi traces
        for label_value, mask in tqdm(
            masks.items(), desc=f"Extracting Traces from Well {pos_name}"
        ):
            # calculate the mean trace for the roi
            masked_data = stack[:, mask]

            # compute the mean trace for each frame
            roi_trace = cast(np.ndarray, masked_data.mean(axis=1))

            # if choosing the top fitted curve
            roi_exponential_decay = _get_exponential_decay(roi_trace, 0.95)
            if roi_exponential_decay and roi_exponential_decay[0] is not None:
                top_exponential_decay = max(roi_exponential_decay,
                                            top_exponential_decay,
                                            key=lambda x: x[2])

            # compute the area of the masksed cells
            roi_size_pixel = masked_data.shape[1]
            roi_size_um = _cell_size_in_um(roi_size_pixel, binning, pixel_size,
                                                objective, magnification)
            if roi_size_um < 10:
                small_rois.append(label_value)
                continue

            condition_1 = condition_2 = None
            if pm_data:
                well_name = pos_name.split("_")[0]
                if well_name in pm_data:
                    condition_1 = pm_data[well_name].get("condition_1")
                    condition_2 = pm_data[well_name].get("condition_2")
            # store the analysis data
            analysis_data[pos_name][str(label_value)] = ROIData(
                raw_trace=roi_trace.tolist(),
                # use_for_bleach_correction=exponential_decay,
                cell_size=roi_size_um,
                condition_1=condition_1,
                condition_2=condition_2,
            )

        avg_r_squared = avg_exponential_decay[2] if (avg_exponential_decay and
            avg_exponential_decay[0] is not None) else 0
        top_r_squared = top_exponential_decay[2] if (top_exponential_decay and
            top_exponential_decay[0] is not None) else 0
        exponential_decay = avg_exponential_decay if (
            abs(avg_r_squared-top_r_squared)<0.01) else (top_exponential_decay)

        i = 1
        while exponential_decay[0] is None:
            exponential_decay = _get_exponential_decay(average_trace, 0.98-i*0.01)
            i += 1

        fitted_curve = exponential_decay[0]
        popts = exponential_decay[1]

        for label_value in tqdm(
            labels_range, desc=f"Performing Bleaching Correction for Well {pos_name}"
            ):
            if label_value in small_rois:
                continue

            roi_data = analysis_data[pos_name][str(label_value)] # for one ROI

            roi_trace = roi_data.raw_trace

            if roi_trace is None:
                continue
            active: bool = True

            # calculate the bleach corrected trace
            bleach_corrected = (
                np.array(roi_trace) - fitted_curve + popts[2]
            )
            # calculate the dF/F TODO: how to calculate F0?
            # F0 = np.min(bleach_corrected)
            # dff = (bleach_corrected - F0) / F0
            dff = calculate_dff(bleach_corrected)
            d_dff, _, _, _, _= deconvolve(dff,  g=(None,None), penalty=1)
            prominence = np.mean(d_dff) * 0.2
            # find the peaks in the bleach corrected trace
            peaks = _find_peaks(d_dff, prominence=prominence) # for one ROI

            if len(peaks) < 2:
                continue

            # Peaks
            amplitudes, start, end, new_peaks = _get_amplitude(d_dff, peaks)

            if new_peaks is None or len(new_peaks) < 2:
                continue

            # max_slopes = self._get_max_slope(d_dff, new_peaks, start)
            rise_time = _get_rise_time(d_dff, amplitudes, new_peaks, start, framerate)
            decay_time = _get_decay_time(new_peaks, end, framerate)

            #ROIData
            iei = _get_iei(new_peaks, framerate)
            mean_iei = np.mean(iei)
            mean_iei_stdev = np.std(iei)
            mean_amplitude = np.mean(amplitudes)
            mean_amplitude_stdev = np.std(amplitudes)
            frequency = len(new_peaks) / (recording_time) # events per second
            mean_rise_time = np.mean(rise_time)
            mean_rise_time_stdev = np.std(rise_time)
            mean_decay_time = np.mean(decay_time)
            mean_decay_time_stdev = np.std(decay_time)
            # mean_max_slope = np.mean(max_slopes)
            # mean_max_slope_stdev = np.std(max_slopes)
            # store the analysis data
            update = roi_data.replace(
                average_photobleaching_fitted_curve=fitted_curve,
                average_popts=popts,
                activity=active,
                bleach_corrected_trace=bleach_corrected.tolist(),
                peaks=[Peaks(peak=new_peaks[i],
                             amplitude=amplitudes[i],
                            #  max_slope=max_slopes[i],
                             rise_time=rise_time[i],
                             decay_time=decay_time[i],
                             start=start[i],
                             end=end[i]
                             ) for i in range(len(new_peaks))],
                mean_amplitude=mean_amplitude,
                mean_amplitude_stdev=mean_amplitude_stdev,
                frequency=frequency,
                mean_rise_time=mean_rise_time,
                mean_rise_time_stdev=mean_rise_time_stdev,
                mean_decay_time=mean_decay_time,
                mean_decay_time_stdev=mean_decay_time_stdev,
                mean_iei=mean_iei,
                mean_iei_stdev=mean_iei_stdev,
                # mean_max_slope=mean_max_slope,
                # mean_max_slope_stdev=mean_max_slope_stdev,
                dff=dff.tolist(),
                d_dff=d_dff.tolist()
            )
            analysis_data[pos_name][str(label_value)] = update

        path = Path(output_path) / f"{pos_name}.json"
        with path.open("w") as file:
            json.dump(
                analysis_data[pos_name],
                file,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )
    return analysis_data

def _get_fov_name(event_key: str, meta: list[dict], p: int) -> str:
    """Retrieve the fov name from metadata."""
    # the "Event" key was used in the old metadata format
    pos_name = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
    return f"{pos_name}_p{p}"

def _get_labels_file_for_position(fov: str, p: int) -> str | None:
    """Retrieve the labels file for the given position."""
    # if the fov name does not end with "_p{p}", add it
    _failed_labels = []
    labels_name = f"{fov}.tif" if fov.endswith(f"_p{p}") else f"{fov}_p{p}.tif"
    labels_path = _get_labels_file(labels_name)
    if labels_path is None:
        _failed_labels.append(labels_name)
        print(f"No labels found for {labels_name}!")
    return labels_path, _failed_labels

def _get_labels_file(_labels_path: str, label_name: str) -> str | None:
    """Get the labels file for the given name."""
    for label_file in Path(_labels_path).glob("*.tif"):
        if label_file.name.endswith(label_name):
            return str(label_file)
    return None

def _create_label_masks_dict(labels: np.ndarray) -> dict:
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
