"""A Micro-Manager GUI based on pymmcore-widgets and pymmcore-plus."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"

import pymmcore_plus

pymmcore_plus.configure_logging("D:/pymmcore-plus-log/pymmcore-plus.log")

from ._main_window import MicroManagerGUI

__all__ = ["MicroManagerGUI"]
