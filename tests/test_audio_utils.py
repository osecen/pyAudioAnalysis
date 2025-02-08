import pytest
import streamlit as st
from datetime import datetime
import logging
from pyAudioAnalysis.audio_utils import st_debug

@pytest.fixture
def setup_streamlit():
    # Clear session state before each test
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.debug_enabled = True

@pytest.fixture
def setup_logger():
    # Setup a test logger
    logger = logging.getLogger('pyAudioAnalysis')
    logger.setLevel(logging.DEBUG)
    return logger

def test_st_debug_ui_logging(setup_streamlit):
    """Test that debug messages are correctly added to session state."""
    # Test debug message
    test_message = "Test debug message"
    st_debug(test_message, level='debug')
    
    # Verify debug logs exist in session state
    assert 'debug_logs' in st.session_state
    assert len(st.session_state.debug_logs) == 1
    
    # Verify log entry structure
    log_entry = st.session_state.debug_logs[0]
    assert 'timestamp' in log_entry
    assert 'level' in log_entry
    assert 'message' in log_entry
    assert log_entry['message'] == test_message
    assert log_entry['level'] == 'debug'

def test_st_debug_duplicate_messages(setup_streamlit):
    """Test that duplicate messages are not added."""
    test_message = "Duplicate message"
    
    # Add same message twice
    st_debug(test_message, level='debug')
    st_debug(test_message, level='debug')
    
    # Verify only one instance is logged
    assert len(st.session_state.debug_logs) == 1

def test_st_debug_different_levels(setup_streamlit):
    """Test logging with different levels."""
    levels = ['debug', 'info', 'warning', 'error']
    
    # Log a message with each level
    for level in levels:
        st_debug(f"Test {level} message", level=level)
    
    # Verify all messages are logged
    assert len(st.session_state.debug_logs) == len(levels)
    
    # Verify levels are correctly stored
    logged_levels = [log['level'] for log in st.session_state.debug_logs]
    assert logged_levels == levels

def test_st_debug_disabled(setup_streamlit):
    """Test that logging is disabled when debug_enabled is False."""
    st.session_state.debug_enabled = False
    
    st_debug("This should not be logged")
    
    # Verify no logs were added
    assert 'debug_logs' not in st.session_state

@pytest.mark.e2e
def test_st_debug_file_logging(setup_streamlit, setup_logger, caplog):
    """Test that messages are correctly logged to file."""
    caplog.set_level(logging.DEBUG)
    
    test_message = "Test file logging"
    st_debug(test_message, level='debug')
    
    # Verify message appears in logs
    assert test_message in caplog.text
    
    # Test different levels
    levels = ['debug', 'info', 'warning', 'error']
    for level in levels:
        message = f"Test {level} message"
        st_debug(message, level=level)
        assert message in caplog.text
