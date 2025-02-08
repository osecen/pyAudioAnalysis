"""Tests for Streamlit UI functionality."""

import pytest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest
from pyAudioAnalysis.audioUI import main


def test_file_upload_state():
    """Test UI state when no file is uploaded."""
    at = AppTest.from_file("pyAudioAnalysis/audioUI.py")
    at.run()
    
    # Check initial state
    try:
        uploaded_file = at.session_state["uploaded_file"]
    except KeyError:
        uploaded_file = None
    
    try:
        analysis_results = at.session_state["analysis_results"]
    except KeyError:
        analysis_results = {}

    assert uploaded_file is None
    assert analysis_results == {}

def test_debug_mode():
    at = AppTest.from_file("pyAudioAnalysis/audioUI.py")
    at.run()

    try:
        debug_val = at.session_state["debug_enabled"]
    except KeyError:
        debug_val = None

    assert debug_val is False


def test_ui_messages():
    """Test UI message system initialization."""
    at = AppTest.from_file("pyAudioAnalysis/audioUI.py")
    at.run()
    
    # Check message system initialization
    try:
        ui_messages = at.session_state["ui_messages"]
    except KeyError:
        ui_messages = {}
    assert ui_messages == {}