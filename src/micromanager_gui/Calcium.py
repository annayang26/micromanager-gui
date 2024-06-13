import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
from datetime import date
from classes.cp_seg import SegmentNeurons
from classes.analyze import AnalyzeNeurons
from classes.plotData import PlotData

import tifffile as tff
import numpy as np

from PyQt6.QtWidgets import (QApplication, QLabel, QDialog,
                             QGridLayout, QPushButton, QFileDialog,
                             QLineEdit, QCheckBox)
from _readers._tensorstore_zarr_reader import TensorstoreZarrReader

class Calcium(QDialog):
    """Test different ways to find spikes in Calcium Recordings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection: str = None
        self.setWindowTitle("Calcium")

        # select the folder to analyze/reanalyze
        self._select_btn = QPushButton("Choose")
        self._select_btn.clicked.connect(self._select_folder)
        self.folder_c = QLabel("")

        self._check_seg = QCheckBox(text="Segment")
        self._check_ana = QCheckBox(text="Analyze")

        filename = QLabel("Enter filename/date: ")
        self.fname = QLineEdit('TEST')

        self._plot_btn = QPushButton("Plot")
        self._plot_btn.clicked.connect(self._plot_data)

        self.ok_btn = QPushButton("Run")
        self.ok_btn.clicked.connect(self._run)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)

        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Select the folder: "), 0, 0)
        self.layout.addWidget(self._select_btn, 0, 1)
        self.layout.addWidget(self.folder_c, 1, 0)

        self.layout.addWidget(self._check_seg, 2, 0)
        self.layout.addWidget(self._check_ana, 2, 1)

        self.layout.addWidget(filename, 5, 0)
        self.layout.addWidget(self.fname, 5, 1)
        self.layout.addWidget(self.ok_btn, 6, 0)
        self.layout.addWidget(self.cancel_btn, 6, 1)
        self.layout.addWidget(self._plot_btn, 8, 0)

        self.setLayout(self.layout)

        self.seg: SegmentNeurons = None
        self.analysis: AnalyzeNeurons = None
        self.plot_data: PlotData = PlotData()
        self.folder_list: list = []
        self._model_loaded: bool = False

    def _update_fname(self, file: str) -> None:
        '''Update the filename displayed on the selection window.'''
        _, fname = os.path.split(file)
        self.folder_c.setText(fname)
        self.layout.addWidget(self.folder_c, 1, 0)

    def _select_folder(self) -> None:
        """Select folder that contains dff.csv file."""
        dlg = QFileDialog(self)
        # dlg.setFileMode(ExistingFiles)
        self.folder_path = dlg.getExistingDirectory(None, "Select Folder")
        self._update_fname(self.folder_path)

    def _load_module(self, img_size: int):
        """Load Segmentation or Analyze."""
        if self._check_seg.isChecked():
            self.seg = SegmentNeurons()
            self.seg._load_model(img_size)
            self._model_loaded = True

        if self._check_ana.isChecked():
            self.analysis = AnalyzeNeurons()

    def _run(self):
        """Run."""
        today = date.today().strftime("%y%m%d")
        additional_name = f"_{today}_{self.fname.text()}"

        if not self.folder_path.endswith("tensorstore.zarr"):
            for folder_path, _, _ in os.walk(self.folder_path):
                if folder_path.endswith("tensorstore.zarr"):
                    folder_name = folder_path

        folder_name = self.folder_path
        r = TensorstoreZarrReader(folder_name)
        folder_path = r.path
        print(f"    folder path: {folder_path}")
        r_shape = r.store.shape
        total_pos = r_shape[0]

        all_pos = r.sequence.stage_positions
        self.framerate, self.binning, self.pixel_size, self.objective, self.magnification = self.extract_metadata(r)

        for pos in range(total_pos):
            rec = r.isel({'p': pos}) # shape(n frames, x, y)

            self.img_stack = rec
            self.img_path = folder_path
            self.img_name = all_pos[pos].name

            img_size = rec.shape[-1]
            self.img_size = img_size

            print(f'           Analyzing {self.img_name} at pos {pos}. shape: {rec.shape}')
            if not self._model_loaded:
                self._load_module(img_size)
                print("___________________Loading model")

            save_path = os.path.join(folder_path, self.img_name)
            save_path = save_path + additional_name

            t_frame = rec.shape[0]
            # if segmentation, run segmentation
            if self.seg and self.seg._check_model():
                self.roi_dict, self.raw_signal, self.dff = self.seg._run(rec, save_path)
                self.analysis.analyze(self.roi_dict, None, self.raw_signal, save_path, t_frame, self.framerate,
                                      self.binning, self.pixel_size, self.objective, self.magnification,
                                      method="mean", frame_window_ptg=1, prom_pctg=0.25)

        _ = self.analysis.compile_files(folder_name, "_compiled.csv", None, additional_name)
        # if len(self.folder_list) > 0:
        #     self.compile_data(self.folder_list[-1], "summary.txt", None, "_compiled.csv")
        #     del self.folder_list[-1]

        print("------------FINISHED-------------")
        self.analysis: AnalyzeNeurons = None
        self.seg = None
        self.folder_list: list = []

    def extract_metadata(self, r: TensorstoreZarrReader)->float:
        '''
        extract information from the metadata
        '''
        exposure = r.sequence.channels[0].exposure
        framerate = 1 / exposure

        binning = int(2048 / 1024)

        pixel_size = 0.325

        objective = 20

        magnification = 1

        return framerate, binning, pixel_size, objective, magnification

    def _record_folders(self, folder: str):
        """Record folder location for compilation."""
        if folder not in self.folder_list:
            self.folder_list.append(folder)

    def _plot_data(self):
        """Plot data."""
        # if self.group_name == "opional":
        #     self.group_name = ""
        
        # groups = self.group_name.text().split(",")
        print("Plotting")
        self.plot_data.just_plot(self.folder_path)

    def _compile_plot(self, base_folder: str, csv_name: str, evk: str | None, groups: list[str]):
        """To plot after compile."""
        for group in groups:
            compile_name = base_folder + group + csv_name
            csv_path = os.path.join(base_folder, compile_name)
            self.plot_data.ana_plot(csv_path, evk, group)

class InputGroup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input group name: ")

if __name__ == "__main__":
    sd_app = QApplication(sys.argv)
    sd = Calcium()
    sd.show()
    sys.exit(sd_app.exec())

