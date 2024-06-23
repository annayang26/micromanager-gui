from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import Peaks, ROIData, _ElapsedTimer, _WaitingProgressBar, show_error_dialog

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
    from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


logger = logging.getLogger("analysis_logger")
logger.setLevel(logging.DEBUG)
log_file = Path(__file__).parent / "analysis_logger.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def single_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.array(a * np.exp(-b * x) + c)


class _SelectAnalysisPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
    ) -> None:
        super().__init__(parent, label, path, tooltip)

    def _on_browse(self) -> None:
        dialog = QFileDialog(self, f"Select the {self._label_text}.")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dialog.setDirectory(self._current_path)

        if dialog.exec() == QFileDialog.Accepted:
            selected_path = dialog.selectedFiles()[0]
            self._path.setText(selected_path)


class _AnalyseCalciumTraces(QWidget):
    progress_bar_updated = Signal()

    def __init__(
        self,
        parent: PlateViewer | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
        labels_path: str | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer | None = parent

        self._data: TensorstoreZarrReader | OMEZarrReader | None = data

        self._labels_path: str | None = labels_path

        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        self._worker: GeneratorWorker | None = None

        self._cancelled: bool = False

        self._output_path = _SelectAnalysisPath(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
        )

        progress_wdg = QWidget(self)
        progress_wdg_layout = QHBoxLayout(progress_wdg)
        progress_wdg_layout.setContentsMargins(0, 0, 0, 0)

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.clicked.connect(self.cancel)

        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        progress_wdg_layout.addWidget(self._run_btn)
        progress_wdg_layout.addWidget(self._cancel_btn)
        progress_wdg_layout.addWidget(self._progress_bar)
        progress_wdg_layout.addWidget(self._progress_pos_label)
        progress_wdg_layout.addWidget(self._elapsed_time_label)

        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        self.progress_bar_updated.connect(self._update_progress_bar)

        self.groupbox = QGroupBox("Extract Traces", self)
        self.groupbox.setCheckable(True)
        self.groupbox.setChecked(False)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._output_path)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

        self._saving_waiting_bar = _WaitingProgressBar(text="Saving Analysis Data")
        self._cancel_waiting_bar = _WaitingProgressBar(text="Stopping all the Tasks...")

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader) -> None:
        self._data = data

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, labels_path: str) -> None:
        self._labels_path = labels_path

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        return self._analysis_data

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        pos = self._prepare_for_running()

        if pos is None:
            return

        logger.info("Number of positions: %s", pos)

        self._progress_bar.reset()
        self._progress_bar.setRange(0, pos)
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText(f"[0/{self._progress_bar.maximum()}]")

        # start elapsed timer
        self._elapsed_timer.start()

        self._cancelled = False

        self._worker = create_worker(
            self._extract_traces,
            positions=pos,
            _start_thread=True,
            _connect={"finished": self._on_worker_finished},
        )

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is None or not self._worker.is_running:
            return

        self._cancelled = True

        self._worker.quit()

        self._cancel_waiting_bar.start()

        # stop the elapsed timer
        self._elapsed_timer.stop()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def _save_analysis_data(self, path: str | Path) -> None:
        """Save the analysis data to a JSON file in a separate thread."""
        # temporarily disable the whole widget
        if self._plate_viewer is not None:
            self._plate_viewer.setEnabled(False)

        # start the waiting progress bar
        self._saving_waiting_bar.start()

        create_worker(
            self._save_as_json,
            path=path,
            _start_thread=True,
            _connect={
                "finished": self._on_saving_finished,
                "errored": self._on_saving_finished,
            },
        )

    def _on_saving_finished(self) -> None:
        """Called when the saving is finished."""
        self._saving_waiting_bar.stop()
        # re-enable the whole widget
        if self._plate_viewer is not None:
            self._plate_viewer.setEnabled(True)

    def _prepare_for_running(self) -> int | None:
        """Prepare the widget for running.

        Returns the number of positions or None if an error occurred.
        """
        if self._worker is not None and self._worker.is_running:
            return None

        if self._data is None or self._labels_path is None:
            logger.error("No data or labels path provided!")
            show_error_dialog(self, "No data or labels path provided!")
            return None

        sequence = self._data.sequence
        if sequence is None:
            logger.error("No useq.MDAsequence found!")
            show_error_dialog(self, "No useq.MDAsequence found!")
            return None

        if path := self._output_path.value():
            save_path = Path(path)
            if save_path.is_dir():
                name = self._get_save_name()
                save_path = save_path / f"{name}.json"
            elif not save_path.suffix:
                save_path = save_path.with_suffix(".json")
            self._output_path.setValue(str(save_path))
            # check if the parent directory exists
            if not save_path.parent.exists():
                logger.error("Output Path does not exist!")
                show_error_dialog(self, "Output Path does not exist!")
                return None
        else:
            logger.error("No Output Path provided!")
            show_error_dialog(self, "No Output Path provided!")
            return None

        return len(sequence.stage_positions)

    def _get_save_name(self) -> str:
        """Generate a save name based on metadata."""
        name = "analysis_data"
        if self._data is not None:
            seq = self._data.sequence
            if seq is not None:
                meta = seq.metadata.get(PYMMCW_METADATA_KEY, {})
                name = meta.get("save_name", name)
                name = f"{name}_analysis_data"
        return name

    def _save_as_json(self, path: str | Path) -> None:
        """Save the analysis data to a JSON file."""
        logger.info("Saving analysis data to %s", path)

        if isinstance(path, str):
            path = Path(path)
        try:
            with path.open("w") as f:
                json.dump(
                    self._analysis_data,
                    f,
                    default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                    indent=2,
                )
                f.flush()  # Flush the internal buffer to the OS buffer
                os.fsync(f.fileno())  # Ensure the OS buffer is flushed to disk
                logger.info("Analysis data saved to %s", path)

        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            show_error_dialog(self, f"An unexpected error occurred: {e}")

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._output_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)

    def _on_worker_finished(self) -> None:
        """Called when the extraction is finished."""
        logger.info("Extraction of traces finished.")

        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

        # save the analysis data
        if not self._cancelled:
            self._save_analysis_data(self._output_path.value())

        # update the analysis data of the plate viewer
        if self._plate_viewer is not None:
            self._plate_viewer.analysis_data = self._analysis_data
            self._plate_viewer._analysis_file_path = self._output_path.value()

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._elapsed_time_label.setText(time_str)

    def _update_progress_bar(self) -> None:
        """Update the progress bar value."""
        if self._check_for_abort_requested():
            return
        value = self._progress_bar.value() + 1
        self._progress_bar.setValue(value)
        self._progress_pos_label.setText(f"[{value}/{self._progress_bar.maximum()}]")

    def _get_labels_file(self, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        if self._labels_path is None:
            return None
        for label_file in Path(self._labels_path).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    def _check_for_abort_requested(self) -> bool:
        return bool(self._worker is not None and self._worker.abort_requested)

    def _extract_traces(self, positions: int) -> None:
        """Extract the roi traces in multiple threads."""
        logger.info("Starting traces extraction...")

        cpu_count = os.cpu_count() or 1
        chunk_size = max(1, positions // cpu_count)

        logger.info("CPU count: %s", cpu_count)
        logger.info("Chunk size: %s", chunk_size)

        try:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(
                        self._extract_trace_for_chunk,
                        start,
                        min(start + chunk_size, positions),
                    )
                    for start in range(0, positions, chunk_size)
                ]

                for idx, future in enumerate(as_completed(futures)):
                    if self._check_for_abort_requested():
                        logger.info("Abort requested, cancelling all futures.")
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                        logger.info(f"Chunk {idx + 1} completed.")
                    except Exception as e:
                        logger.error("An error occurred in a chunk: %s", e)
                        show_error_dialog(self, f"An error occurred in a chunk: {e}")
                        break

            logger.info("All tasks completed.")

        except Exception as e:
            logger.error("An error occurred: %s", e)
            show_error_dialog(self, f"An error occurred: {e}")

    def _extract_trace_for_chunk(self, start: int, end: int) -> None:
        """Extract the roi traces for the given chunk."""
        for p in range(start, end):
            if self._check_for_abort_requested():
                break
            self._extract_trace_per_position(p)

    def _extract_trace_per_position(self, p: int) -> None:
        """Extract the roi traces for the given position."""
        if self._data is None or self._check_for_abort_requested():
            return

        data, meta = self._data.isel(p=p, metadata=True)

        # get position name from metadata
        well = meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")

        # create the dict for the well
        if well not in self._analysis_data:
            self._analysis_data[well] = {}

        # matching label name
        labels_name = f"{well}_p{p}.tif"

        # get the labels file
        labels = tifffile.imread(self._get_labels_file(labels_name))
        if labels is None:
            logger.error("No labels found for %s!", labels_name)
            show_error_dialog(self, f"No labels found for {labels_name}!")
            return

        # get the range of labels
        labels_range = range(1, labels.max() + 1)

        # create masks for each label
        masks = {label_value: (labels == label_value) for label_value in labels_range}

        logger.info("Processing well %s", well)

        # temporary storage for trace to use for photobleaching correction
        fitted_curves: list[tuple[list[float], list[float], float]] = []

        roi_trace: np.ndarray | list[float] | None

        # extract roi traces
        logger.info(f"Extracting Traces from Well {well}.")
        for label_value, mask in tqdm(
            masks.items(), desc=f"Extracting Traces from Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            # calculate the mean trace for the roi
            masked_data = data[:, mask]

            # compute the mean for each frame
            roi_trace = cast(np.ndarray, masked_data.mean(axis=1))

            # calculate the exponential decay for photobleaching correction
            exponential_decay = self._get_exponential_decay(roi_trace)
            if exponential_decay is not None:
                fitted_curves.append(exponential_decay)

            # store the analysis data
            self._analysis_data[well][str(label_value)] = ROIData(
                raw_trace=roi_trace.tolist(),
                use_for_bleach_correction=exponential_decay,
            )

        # average the fitted curves
        logger.info(f"Averaging the fitted curves well {well}.")
        popts = np.array([popt for _, popt, _ in fitted_curves])
        average_popts = np.mean(popts, axis=0)
        time_points = np.arange(data.shape[0])
        average_fitted_curve = single_exponential(time_points, *average_popts)

        # perform photobleaching correction
        logger.info(f"Performing Belaching Correction for Well {well}.")
        for label_value in tqdm(
            labels_range, desc=f"Performing Belaching Correction for Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            data = self._analysis_data[well][str(label_value)]

            roi_trace = data.raw_trace

            if roi_trace is None:
                continue

            # calculate the bleach corrected trace
            bleach_corrected = np.array(roi_trace) / average_fitted_curve

            # F0 = np.mean(bleach_corrected[:10])
            # dff = (bleach_corrected - F0) / F0

            # find the peaks in the bleach corrected trace
            peaks = self._find_peaks(bleach_corrected, 0.1)
            # store the analysis data
            update = data.replace(
                average_photobleaching_fitted_curve=average_fitted_curve.tolist(),
                bleach_corrected_trace=bleach_corrected.tolist(),
                peaks=[Peaks(peak=peak) for peak in peaks],
                # dff=dff.tolist(),
            )
            self._analysis_data[well][str(label_value)] = update

        # update the progress bar
        self.progress_bar_updated.emit()

    def _smooth_and_normalize(self, trace: np.ndarray) -> np.ndarray:
        """Smooth and normalize the trace between 0 and 1."""
        # smoothing that preserves the peaks
        smoothed = savgol_filter(trace, window_length=5, polyorder=2)
        # normalize the smoothed trace from 0 to 1
        return cast(
            np.ndarray,
            (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed)),
        )

    def _get_exponential_decay(
        self, trace: np.ndarray
    ) -> tuple[list[float], list[float], float] | None:
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
            logger.error("Error fitting curve: %s", e)
            return None

        return (
            None
            if r_squared <= 0.98
            else (fitted_curve.tolist(), popt.tolist(), float(r_squared))
        )

    def _find_peaks(
        self, trace: np.ndarray, prominence: float | None = None
    ) -> list[int]:
        """Smooth the trace and find the peaks."""
        smoothed_normalized = self._smooth_and_normalize(trace)
        # find the peaks # TODO: find the necessary parameters to use
        peaks, _ = find_peaks(smoothed_normalized, width=3, prominence=prominence)
        peaks = cast(np.ndarray, peaks)
        return cast(list[int], peaks.tolist())
