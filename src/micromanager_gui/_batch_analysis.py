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
import xlsxwriter
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from pymmcore_widgets.mda._save_widget import OME_ZARR, WRITERS, ZARR_TESNSORSTORE
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QGridLayout,
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

from ._plate_viewer._plate_map import PlateMapData
from ._plate_viewer._util import GENOTYPE_MAP, GREEN, RED, TREATMENT_MAP, Peaks, ROIData
from ._readers._ome_zarr_reader import OMEZarrReader
from ._readers._tensorstore_zarr_reader import TensorstoreZarrReader

if TYPE_CHECKING:
    from threading import Event

    from superqt.utils import FunctionWorker

EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])
FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
DATA_KEY = 0
LABELS = "_labels"

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
            settings_groupbox_layout.addWidget(buttons_wdg, 3, 0, 2, 1)
            # settings_groupbox_layout.addWidget(buttons_wdg, 3, 0, 2, 1)

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(self.groupbox)
            main_layout.addStretch(1)

    def run(self) -> None:
        self._stop_event.clear()
        self._run_worker = create_worker(self._run, _start_thread=True)

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
        input_path = self._input_path.value()
        # input_path = r'/Volumes/Expansion/test'

        if not input_path:
            return

        self._load_plate_map()
        self._handle_plate_map()

        recording_file_path = []
        labels_path = []
        for folder in Path(input_path).iterdir():
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.name.endswith(EXT):
                        recording_file_path.append(str(f))
                    if f.name.endswith(LABELS):
                        labels_path.append(str(f))

                # TODO: uncomment the following line when running actual analysis
                if len(recording_file_path) != len(labels_path):
                    print(
                        f"{recording_file_path[-1]} is not segmented. Please run segmentaiton first."  # noqa: E501
                        )
                    self.cancel()
                    break

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpu_count
                ) as executor:
                for f, label in zip(recording_file_path, labels_path):
                    futures = executor.submit(_analyze_data, f, label,
                                            self._plate_map_data, self._genotype_pm,
                                            self._treatment_pm, self._stop_event)
                    self._futures.append(futures)

                    for future in tqdm(
                        concurrent.futures.as_completed(self._futures),
                        total=len(recording_file_path),
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
                self._plate_map_data[data.name] = {"condition_1": data.condition}

            for data in self._treatment_pm:
                if data.name in self._plate_map_data:
                    self._plate_map_data[data.name]["condition_2"] = data.condition
                else:
                    self._plate_map_data[data.name] = {"condition_2": data.condition}

    def _setValue(self, value: Path | str):
        if isinstance(value, (Path, str)):
            with open(value) as pmap:
                data = json.load(pmap)
            value = cast(list, data)

        pm_data_list = []
        for data in value:
            # convert the data to a PlateMapData object if it is a list of strings
            if not isinstance(data, PlateMapData):
                data = PlateMapData(*data)
            pm_data_list.append(data)
        return pm_data_list


def _analyze_data(data_path: str, label_path: str,
                   pm_data: dict[str, dict[str, str]],
                   geno_pm: list[PlateMapData], cond_pm: list[PlateMapData],
                   stop_event: Event) -> None:
    if stop_event.is_set():
        print(f"Analysis process stopped for {data_path}")
        return

    path = Path(data_path)
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

    _save_plate_map(geno_pm=geno_pm,
                    cond_pm=cond_pm,
                    path=path)

    analysis_data = _analyze(
        data=data,
        labels_path=label_path,
        pm_data=pm_data,
        output_path=path,
        positions=positions,
        stop_event=stop_event
    )

    output_csv(output_path=path,
               analysis_data=analysis_data,
               pm_data=pm_data,
               )

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

        # get position name from metadata
        pos_name = meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")

        if pos_name not in analysis_data:
            analysis_data[pos_name] = {}

        total_frames = stack.shape[0]
        binning, magnification, pixel_size, objective,\
            exposure, framerate = _extract_metadata(meta) # exposure time in ms
        framerate *= 1000 # seconds
        recording_time = total_frames/framerate # in seconds

        # matching label name
        label_name = f"{pos_name}_p{p}.tif"
        labels = tifffile.imread(_get_labels_file(labels_path, label_name))
        if labels is None:
            print("No labels found for %s!", label_name)
            continue

        # get the range of labels
        labels_range = range(1, labels.max())

        # create masks for each label
        masks = {label_value: (labels == label_value) for label_value in labels_range}

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

def _get_labels_file(labels_path: str, label_name: str) -> str | None:
    """Get the labels file for the given name."""
    if labels_path is None:
        return None
    for label_file in Path(labels_path).glob("*.tif"):
        if label_file.name.endswith(label_name):
            return str(label_file)
    return None

def _extract_metadata(meta: list[dict]) -> tuple[float]:
    """Extract information from metadata."""
    binning = int(meta[0].get('pco_camera-Binning'))
    magnification = float(meta[0].get('IntermediateMagnification-Magnification')[:-1])
    pixel_size = float(meta[0].get('PixelSizeUm'))
    objective = int(meta[0].get('Nosepiece-Label').split(' ')[-1][:-1])
    exposure = float(meta[0].get('Event').get('exposure'))
    framerate = 1 / exposure

    return binning, magnification, pixel_size, objective, exposure, framerate

def _get_exponential_decay(
    trace: np.ndarray, cut_off: float = 0.98
) -> tuple[list[float], list[float], float]:
    """Fit an exponential decay to the trace.

    Returns None if the R squared value is less than 0.9.
    """
    time_points = np.arange(len(trace))
    initial_guess = [max(trace), 0.01, min(trace)]
    try:
        popt, _ = curve_fit(
            single_exponential, time_points, trace, p0=initial_guess, maxfev=2000
        )
        fitted_curve = single_exponential(time_points, *popt)
        residuals = trace - fitted_curve
        r, _ = pearsonr(trace, fitted_curve)
        ss_total = np.sum((trace - np.mean(trace)) ** 2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - (ss_res / ss_total)
    except Exception as e:
        print("Error fitting curve: %s", e)
        return None

    return (
        (None, None, None)
        if r_squared <= cut_off
        else (fitted_curve.tolist(), popt.tolist(), float(r_squared))
    )

def single_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.array(a * np.exp(-b * x) + c)

def _cell_size_in_um(cell_size_pixel: int, binning: int,
                        pixel_size: float, objective: int, magnification: float
                        )-> int:
    """Convert the cell size in pixel to um."""
    cell_size_um = cell_size_pixel * binning * pixel_size / (
        objective * magnification
    )

    return cell_size_um

def calculate_dff(pc_trace: np.ndarray) -> list[float]:
    dff = []
    bg, _ = _calculate_bg(pc_trace, 100)
    bg = list(bg)
    dff = (pc_trace - bg)/bg
    dff = dff - np.min(dff)

    return dff

def _calculate_bg(f: np.ndarray, window: int) -> tuple[np.ndarray, list[float]]:
    background = np.zeros_like(f)
    background[0] = f[0]
    median = [background[0]]
    for y in range(1, len(f)):
        x = y - window
        if x < 0:
            x = 0
        lower_quantile = f[x:y] <= np.median(f[x:y])
        background[y] = np.mean(f[x:y][lower_quantile])
        median.append(np.median(f[x:y]))
    return background, median

def _find_peaks(
    trace: np.ndarray, prominence: float | None = None
) -> list[int]:
    """Smooth the trace and find the peaks."""
    smoothed_normalized = _smooth_and_normalize(trace)
    peaks, _ = find_peaks(smoothed_normalized, width=3, prominence=prominence)
    peaks = cast(np.ndarray, peaks)
    return cast(list[int], peaks.tolist())

def _smooth_and_normalize(trace: np.ndarray) -> np.ndarray:
    """Smooth and normalize the trace between 0 and 1."""
    # smoothing that preserves the peaks
    smoothed = savgol_filter(trace, window_length=5, polyorder=2)
    # normalize the smoothed trace from 0 to 1
    return cast(
        np.ndarray,
        (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed)),
    )

def _get_amplitude(dff: list[float], peaks: list[int], deriv_threshold=0.01,
                reset_num=17, neg_reset_num=2, total_dist=40, min_dist=5
                ) -> tuple[list[float], list[int], list[int], list[int]]:
    """Calculate amplitudes, peak indices, and base indices of each ROI."""
    amplitudes = []
    start_indices = []
    end_indices = []
    new_peaks = []

    if len(peaks) < 2:
        return

    dff_deriv = np.diff(dff)
    len_dff_deriv = len(dff_deriv)

    for peak in peaks:
        start_index = peak
        end_index = peak
        under_thresh_count = 0
        total_count = 0

        if start_index >= 0:
            while (start_index >= 0
                    and total_count < total_dist):
                start_index -= 1
                total_count += 1
                if start_index in peaks:
                    negative_count = 0
                    while start_index < len_dff_deriv and\
                            dff_deriv[start_index] < 0 and\
                                negative_count < neg_reset_num:
                        start_index += 1
                        if dff_deriv[start_index] < 0:
                            negative_count += 1
                        else:
                            negative_count = 0
                    break
                if dff_deriv[start_index] < deriv_threshold:
                    under_thresh_count += 1
                else:
                    under_thresh_count = 0
                if under_thresh_count >= reset_num:
                    break

        under_thresh_count = 0
        total_count = 0

        if end_index < len_dff_deriv - 1:
            while (end_index < len_dff_deriv - 1
                    and total_count < total_dist):
                end_index += 1
                total_count += 1
                if end_index in peaks:
                    negative_count = 0
                    while (end_index >= peak
                            and dff_deriv[end_index] > 0
                            and negative_count < neg_reset_num):
                        end_index -= 1
                        if dff_deriv[end_index] > 0:
                            negative_count += 1
                        else:
                            negative_count = 0
                    break
                if dff_deriv[end_index] < deriv_threshold:
                    under_thresh_count += 1
                else:
                    under_thresh_count = 0
                if under_thresh_count >= reset_num:
                    break

        spk_to_end = dff[peak:(end_index + 1)]
        start_to_spk = dff[start_index:peak]
        amplitude = 0

        if len(spk_to_end) < min_dist or len(start_to_spk) < min_dist:
            continue

        f_start_index = int(peak - (len(start_to_spk) -
                                    np.argmin(start_to_spk)))
        f_end_index = int(peak + np.argmin(spk_to_end))

        if (peak - f_start_index < min_dist
            or f_end_index - peak < min_dist):
            continue

        amplitude = dff[peak] - dff[f_start_index]

        if amplitude > 0:
            start_indices.append(f_start_index)
            end_indices.append(f_end_index)
            amplitudes.append(amplitude)
            new_peaks.append(peak)

    return amplitudes, start_indices, end_indices, new_peaks

def _get_max_slope(dff: list[float], peaks: list[int], bases: list[int]
                    ) -> list[float]:
    """Get max slope of each peak in one ROI."""
    max_slopes = []

    if len(peaks) > 0:
        for i in range(len(peaks)):
            peak_index = peaks[i]
            base_index = bases[i]

            slope_window = dff[base_index:(peak_index + 1)]
            slope_window_a = slope_window[:-1]
            slope_window_b = slope_window[1:]
            max_slope = max([b-a for a,b in zip(slope_window_a, slope_window_b)])
            max_slopes.append(max_slope)

    return max_slopes

# FluoroSNNAP uses from base to max_slope point
def _get_rise_time(dff: list[float], amplitude: list[float], peaks: list[int],
                    start: list[int], framerate: float) -> list[float]:
    """Get Raise Time for each peak."""
    rise_time = []
    if not (len(amplitude) == len(peaks) == len(start)):
        raise ValueError("The length of amplitude, peaks, and start lists must be equal.")

    # NOTE: time to reach half of amplitude
    for amp, peak, s in zip(amplitude, peaks, start):
        try:
            limit_range = int((peak + 1 - s)/3)
            if s + limit_range >= peak - limit_range:
                print(f"Invalid range for peak {peak}, start {s}")
                rise_time.append(np.nan)
                continue

            rise_range = dff[s+limit_range:(peak+1)-limit_range]

            if len(rise_range) == 0:
                print(f"Rise range is empty for peak {peak}, start {s}")
                rise_time.append(np.nan)
                continue

            half_amp = amp/2 + dff[s]
            half_amp_idx = np.argmin([abs(signal - half_amp) for signal in rise_range])
            rise_time.append((limit_range+half_amp_idx)/framerate) #s
        except Exception as e:
            print(f'error in rise time calculation, {e}')

    return rise_time

def _get_decay_time( peaks: list[int], end: list[int], framerate: float
                    ) -> list[float]:
    """Get decay time for each peak."""
    decay_time = [((end[i] - peaks[i] + 1)/ framerate) for i in range(len(peaks))]

    return decay_time

# IEI: peak to peak
def _get_iei(peaks: list[int], framerate: float) -> list[float]:
    """Calculate the interevent interval."""
    iei_frames = np.diff(np.array(peaks))
    iei = cast(list, iei_frames/framerate) #s

    return iei

def output_csv(output_path: str,
               analysis_data: dict,
               pm_data: dict,
               col_per_treatment: int = 12) -> None:
    exp_name = Path(output_path).parent.name

    readout_list = ['Average Cell Size', 'Average Amplitude', 'Average Frequency',
                    'Average Rise Time', 'Average IEI', 'Percentage Active']

    compiled_data_list = _compile_readout_data(analysis_data, pm_data)
    compiled_cond = _compile_conditions(pm_data)
    compiled_geno = _compile_genotypes(pm_data)
    if compiled_data_list:
        for readout, readout_data in zip(readout_list, compiled_data_list):
            file_path = Path(output_path)/f"{exp_name}_{readout}.xlsx"
            wkbk = xlsxwriter.Workbook(file_path, {'nan_inf_to_errors': True})
            # with xlsxwriter.Workbook(file_path, {'nan_inf_to_errors': True}) as wkbk:
            wkst = wkbk.add_worksheet(readout)
            num_format = wkbk.add_format({'num_format': '0.00'})
            wkst.write(0, 0, readout)

            # write conditions
            for i, condition in enumerate(compiled_cond):
                for repeat in range(col_per_treatment):
                    wkst.write(0, i*col_per_treatment+repeat+1, condition)

            # write genotypes
            for i, genotype in enumerate(compiled_geno):
                geno = genotype
                if genotype.lower() == "crispr":
                    geno = "+/+"
                elif genotype.lower() == "patient":
                    geno = "+/-"
                elif genotype.lower() == "null":
                    geno = "-/-"

                wkst.write(i+1, 0, geno)

            for genotype, cond_data in readout_data.items():
                    for cond, data_list in cond_data.items():
                        for i in range(col_per_treatment):
                            try:
                                start = compiled_cond.index(cond)
                                row = compiled_geno.index(genotype)+1
                            except ValueError:
                                start = 0
                                row = 5

                            if i < len(data_list):
                                entry = data_list[i]
                                if entry == 'N/A':
                                    wkst.write(row, start*col_per_treatment+i+1, entry)
                                else:
                                    wkst.write_number(row,
                                                        start*col_per_treatment+i+1,
                                                        float(entry),
                                                        num_format)
                            else:
                                entry = 'N/A'
                                wkst.write(row, start*col_per_treatment+i+1, entry)
            wkbk.close()
    else:
        print("No data were found. Please check the plate map and data!")

def _compile_readout_data(
        analysis_data: dict[str, dict[str, ROIData]], pm_data: dict
        ) -> list[dict[str, dict[str, list[float]]]]:
    data_by_metrics = []
    mean_amplitude_dict = {}
    mean_cell_size_dict = {}
    mean_frequency_dict = {}
    # mean_max_slope_dict = {}
    mean_rise_time_dict = {}
    mean_iei_dict = {}
    activity_dict = {}

    data_to_compile = analysis_data
    plate_map_keys = list(pm_data.keys())

    if len(plate_map_keys) > 0:
        for fov, fov_dict in data_to_compile.items():
            well = fov.split('_')[0]
            if well in plate_map_keys:
                genotype = pm_data[well].get("condition_1")
                treatment = pm_data[well].get("condition_2")

                amplitude_list = []
                cell_size_list = []
                frequency_list = []
                iei_list = []
                rise_time_list = []
                active_cells: int = 0

                for roiData in fov_dict.values():
                    if roiData.activity:
                        cell_size_list.append(roiData.cell_size)
                        amplitude_list.append(roiData.mean_amplitude)
                        frequency_list.append(roiData.frequency)
                        iei_list.append(roiData.mean_iei)
                        rise_time_list.append(roiData.mean_rise_time)
                        active_cells += 1

                mean_amplitude_fov = np.nanmean(amplitude_list, dtype=np.float64
                                                ) if (len(amplitude_list)>0
                                                        ) else 'N/A'
                mean_cell_size_fov = np.nanmean(cell_size_list, dtype=np.float64
                                                ) if (len(cell_size_list)>0
                                                        ) else 'N/A'
                mean_frequency_fov = np.nanmean(frequency_list, dtype=np.float64
                                                ) if (len(frequency_list)>0
                                                        ) else 'N/A'
                # mean_max_slope_fov = np.mean(max_slope_list)
                mean_iei_fov = np.nanmean(iei_list, dtype=np.float64
                                                ) if (len(iei_list)>0
                                                        ) else 'N/A'
                mean_rise_time_fov = np.nanmean(rise_time_list, dtype=np.float64
                                                ) if (len(rise_time_list)>0
                                                        ) else 'N/A'
                pctg_active = active_cells / len(list(fov_dict.keys())) * 100 if (
                                                len(cell_size_list)>0
                                                        ) else 'N/A'

                if genotype not in mean_amplitude_dict:
                    mean_amplitude_dict[genotype] = {}
                if treatment not in mean_amplitude_dict[genotype]:
                    mean_amplitude_dict[genotype][treatment] = []
                mean_amplitude_dict[genotype][treatment].append(mean_amplitude_fov)

                if genotype not in mean_cell_size_dict:
                    mean_cell_size_dict[genotype] = {}
                if treatment not in mean_cell_size_dict[genotype]:
                    mean_cell_size_dict[genotype][treatment] = []
                mean_cell_size_dict[genotype][treatment].append(mean_cell_size_fov)

                if genotype not in mean_frequency_dict:
                    mean_frequency_dict[genotype] = {}
                if treatment not in mean_frequency_dict[genotype]:
                    mean_frequency_dict[genotype][treatment] = []
                mean_frequency_dict[genotype][treatment].append(mean_frequency_fov)

                if genotype not in mean_iei_dict:
                    mean_iei_dict[genotype] = {}
                if treatment not in mean_iei_dict[genotype]:
                    mean_iei_dict[genotype][treatment] = []
                mean_iei_dict[genotype][treatment].append(mean_iei_fov)

                if genotype not in mean_rise_time_dict:
                    mean_rise_time_dict[genotype] = {}
                if treatment not in mean_rise_time_dict[genotype]:
                    mean_rise_time_dict[genotype][treatment] = []
                mean_rise_time_dict[genotype][treatment].append(mean_rise_time_fov)

                if genotype not in activity_dict:
                    activity_dict[genotype] = {}
                if treatment not in activity_dict[genotype]:
                    activity_dict[genotype][treatment] = []
                activity_dict[genotype][treatment].append(pctg_active)

        data_by_metrics.append(mean_cell_size_dict)
        data_by_metrics.append(mean_amplitude_dict)
        data_by_metrics.append(mean_frequency_dict)
        data_by_metrics.append(mean_rise_time_dict)
        data_by_metrics.append(mean_iei_dict)
        data_by_metrics.append(activity_dict)
    else:
        print("Data length doesn't match platemap. Will not output CSV files!")
    return (None if len(data_by_metrics) < 1 else data_by_metrics)

def _compile_conditions(pm_data: dict) -> list[str]:
    return list({value["condition_2"] for value in pm_data.values()})

def _compile_genotypes(pm_data: dict) -> list[str]:
    return list({value["condition_1"] for value in pm_data.values()})

def _save_plate_map(geno_pm: list[PlateMapData], cond_pm: list[PlateMapData],
                    path: str) -> None:
    geno_path = Path(path) / GENOTYPE_MAP
    cond_path = Path(path) / TREATMENT_MAP
    with geno_path.open("w") as f1:
        json.dump(geno_pm, f1, indent=2)
    with cond_path.open("w") as f2:
        json.dump(cond_pm, f2, indent=2)
