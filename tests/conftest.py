from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import _mmcore_plus
from qtpy.QtWidgets import QMenuBar

if TYPE_CHECKING:
    from pytest import FixtureRequest
    from qtpy.QtWidgets import QApplication

TEST_CONFIG = str(Path(__file__).parent / "test_config.cfg")


# to create a new CMMCorePlus() for every test
@pytest.fixture(autouse=True)
def global_mmcore():
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration(TEST_CONFIG)
    with patch.object(_mmcore_plus, "_instance", mmc):
        yield mmc


@pytest.fixture()
def _run_after_each_test(request: FixtureRequest, qapp: QApplication):
    """Run after each test to ensure no widgets have been left around.

    When this test fails, it means that a widget being tested has an issue closing
    cleanly. Perhaps a strong reference has leaked somewhere.  Look for
    `functools.partial(self._method)` or `lambda: self._method` being used in that
    widget's code.
    """
    nbefore = len(qapp.topLevelWidgets())
    failures_before = request.session.testsfailed
    yield
    # if the test failed, don't worry about checking widgets
    if request.session.testsfailed - failures_before:
        return
    remaining = qapp.topLevelWidgets()
    # for some reason, in pyside2 and pyside6, when you add a QMenuBar, then when you
    # close the widget you still get a QMenuBar in topLevelWidgets without a parent.
    # Therefore, as a temporary fix, we remove any QMenuBar from the list of remaining
    remaining = [w for w in remaining if not isinstance(w, QMenuBar)]

    print()
    print("______REMAINING______")
    for r in remaining:
        print(r, f"parent: {r.parent()}")

    if len(remaining) > nbefore:
        test = f"{request.node.path.name}::{request.node.originalname}"
        raise AssertionError(f"topLevelWidgets remaining after {test!r}: {remaining}")
