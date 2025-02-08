"""Utility functions for audio processing and model management."""

import os
import numpy as np
import streamlit as st
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioSegmentation as aS
import plotly.graph_objects as go
from datetime import datetime
import logging

def st_debug(message, level='debug'):
    """Log debug messages to sidebar and log file."""
    # Only add logs if debug mode is enabled and message is new
    if st.session_state.get('debug_enabled', False):
        # Initialize debug logs if not present
        if 'debug_logs' not in st.session_state:
            st.session_state.debug_logs = []
        
        # Check if this exact message already exists
        message_exists = any(
            log['message'] == message and log['level'] == level 
            for log in st.session_state.debug_logs
        )
        
        # Only add new messages
        if not message_exists:
            timestamp = datetime.now().strftime('%H:%M:%S')
            # Add new log entry with current timestamp
            st.session_state.debug_logs.append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
            
            # Log to file based on level
            logger = logging.getLogger('pyAudioAnalysis')
            if level == 'debug':
                logger.debug(message)
            elif level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)

def st_message(message, level='info'):
    """
    Store and display messages in the main UI area.
    Messages are stored in session state and will persist until explicitly cleared.
    
    Args:
        message (str): Message to display
        level (str): Message level ('info', 'success', 'warning', 'error')
    """
    # Initialize messages in session state if not present
    if 'ui_messages' not in st.session_state:
        st.session_state.ui_messages = []
    
    # Add message to session state
    message_data = {
        'message': message,
        'level': level,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    st.session_state.ui_messages.append(message_data)

def display_ui_messages():
    """
    Display all stored UI messages.
    Call this function once in your main UI layout where you want messages to appear.
    """
    if 'ui_messages' in st.session_state:
        for msg in st.session_state.ui_messages:
            if msg['level'] == 'info':
                st.info(msg['message'])
            elif msg['level'] == 'success':
                st.success(msg['message'])
            elif msg['level'] == 'warning':
                st.warning(msg['message'])
            elif msg['level'] == 'error':
                st.error(msg['message'])
            else:
                st.write(msg['message'])

def read_audio_file(file_path):
    """
    Read and validate audio file.
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        tuple: (success, sampling_rate, signal)
    """
    try:
        [Fs, x] = audioBasicIO.read_audio_file(file_path)
        """Clear all stored UI messages."""
        if 'ui_messages' in st.session_state:
            st.session_state.ui_messages = []
        if Fs == 0 or x is None:
            st_message("Failed to read audio file", level='error')
            st_debug("Failed to read audio file", level='error')
            return False, None, None
        
        # Add debug info to sidebar
        st_debug(f"Audio loaded - Fs: {Fs}Hz, Duration: {len(x)/Fs:.2f}s")
        
        if len(x.shape) > 1:
            st_message("Multi-channel audio detected - converting to mono", level='warning')
            st_debug("Converting multi-channel to mono", level='warning')
            x = audioBasicIO.stereo_to_mono(x)
        
        return True, Fs, x
    except Exception as e:
        st_message(f"Error reading audio file: {str(e)}", level='error')
        st_debug(f"Error reading audio file: {str(e)}", level='error')
        return False, None, None

def check_model_path(model_path):
    """
    Verify model path exists and is valid.
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        bool: True if model path is valid
    """
    st_debug(f"Checking model path: {model_path}")
    st_debug(f"Absolute path: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        st_message(f"Model not found at: {model_path}", level='error')
        st_debug(f"Model file does not exist: {model_path}", level='error')
        
        # Check parent directory
        parent_dir = os.path.dirname(model_path)
        if os.path.exists(parent_dir):
            st_debug(f"Parent directory exists: {parent_dir}")
            st_debug(f"Contents: {os.listdir(parent_dir)}")
        else:
            st_debug(f"Parent directory does not exist: {parent_dir}", level='error')
        return False
        
    st_debug(f"Model file found at: {model_path}")
    return True

def parse_model_name(model_file):
    """
    Parse model filename to extract metadata.
    
    Args:
        model_file (str): Model filename
        
    Returns:
        tuple: (classifier_type, friendly_name, description, clean_name)
    """
    base_name = os.path.splitext(model_file)[0]
    st_debug(f"Parsing model name: {base_name}")
    
    # ... rest of parse_model_name implementation ...

def ensure_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = os.path.join(os.path.dirname(__file__), "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    st_debug(f"Ensuring models directory exists at: {models_dir}")
    return models_dir

def find_models_directory():
    """
    Find the models directory from possible paths.
    
    Returns:
        str: Path to models directory or None if not found
    """
    # Get the absolute path of the current file (audio_utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(current_dir, "data", "models"),  # pyAudioAnalysis/pyAudioAnalysis/data/models
        os.path.join(current_dir, "..", "data", "models"),  # pyAudioAnalysis/data/models
        os.path.join(os.getcwd(), "pyAudioAnalysis", "data", "models"),  # ./pyAudioAnalysis/data/models
        os.path.join(os.getcwd(), "data", "models"),  # ./data/models
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        st_debug(f"Checking models path: {abs_path}")
        if os.path.exists(abs_path):
            st_debug(f"Found models directory at: {abs_path}")
            return abs_path
            
    st_message("Models directory not found. Checked paths:", level='error')
    for path in possible_paths:
        st_debug(f"- {os.path.abspath(path)}")
    return None

def plot_classification_results(class_names, probabilities):
    """
    Create visualization for classification results.
    
    Args:
        class_names (list): List of class names
        probabilities (list): List of probabilities
    """
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=probabilities,
                text=[f'{p*100:.1f}%' for p in probabilities],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Classification Confidence Scores",
            xaxis_title="Classes",
            yaxis_title="Confidence",
            yaxis_range=[0, 1]
        )
        return fig
    except Exception as e:
        st_message(f"Error creating classification plot: {str(e)}", level='error')
        st_debug(f"Plot creation error: {str(e)}", level='error')
        return None 