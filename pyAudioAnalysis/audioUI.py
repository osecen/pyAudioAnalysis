"""
Streamlit UI for audio analysis.

This module provides a web interface for audio analysis tasks including:
- Classification
- Regression
- Feature Extraction
- Segmentation
- Speaker Diarization
- Beat Extraction

The interface supports WAV file uploads and provides various visualization options
for analysis results.
"""

import streamlit as st
import os
import numpy as np
import tempfile
import traceback
import logging
from datetime import datetime
from pyAudioAnalysis import audio_utils as audio_utils
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioSegmentation import labels_to_segments
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import sys
import platform
import pandas as pd
from scipy.signal import find_peaks
import scipy.io.wavfile as wavfile
import sklearn.cluster
from pyAudioAnalysis.audio_utils import st_debug  # Import the specific function

# Global variables
MODEL_PATHS = {}
# Configure logging
def setup_logging():
    """Configure logging with both file and console handlers."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('pyAudioAnalysis')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # File handler
        log_file = os.path.join(log_dir, f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def parse_model_name(model_file):
    """
    Parse model filename to get classifier type, task, and variant.
    Skip MEANS files as they are handled automatically by the classifier.
    """
    base_name = os.path.splitext(model_file)[0]
    st_debug(f"Parsing model name: {base_name}")
    
    # Skip MEANS files - they are used internally by the classifier
    if base_name.endswith('MEANS'):
        st_debug(f"Skipping MEANS file: {base_name}")
        return None, None, None, None
    
    # Determine classifier type (knn or svm)
    if base_name.startswith('knn'):
        classifier = 'knn'
    elif base_name.startswith('svm'):
        classifier = 'svm_rbf'
    else:
        st_debug(f"Unknown classifier type for: {base_name}")
        return None, None, None, None
    
    # Extract task name
    task = base_name.replace(classifier + '_', '')
    st_debug(f"Extracted task: {task}")
    
    # Create friendly name and description
    if '4class' in task:
        friendly_name = "4-Class Audio Classification"
        description = "Classifies audio into 4 basic classes"
    elif 'movie8class' in task:
        friendly_name = "Movie Genre (8 classes)"
        description = "Classifies movie audio into 8 genres"
    elif 'musical_genre_6' in task:
        friendly_name = "Music Genre (6 classes)"
        description = "Classifies music into 6 different genres"
    elif 'sm' == task.lower():
        friendly_name = "Speech/Music Classification"
        description = "Distinguishes between speech and music"
    elif 'speaker_10' in task:
        friendly_name = "Speaker Recognition (10 speakers)"
        description = "Identifies between 10 different speakers"
    elif 'speaker_male_female' in task:
        friendly_name = "Gender Recognition"
        description = "Classifies speaker gender as male or female"
    else:
        friendly_name = task.replace('_', ' ').title()
        description = f"Classifies audio using {friendly_name}"
    
    st_debug(f"Parsed result: {classifier}, {friendly_name}, {description}")
    return classifier, friendly_name, description, base_name

def check_available_models():
    """Check which models are available in the data/models directory"""
    # Only load models if they haven't been loaded before
    if 'AVAILABLE_MODELS' not in st.session_state:
        st_debug(f"Starting model discovery")
        st_debug(f"Current working directory: {os.getcwd()}")
        
        # Try multiple possible model directory locations
        models_dir = audio_utils.find_models_directory()
        if not models_dir:
            st_message("Could not find models directory", level='error')
            st.session_state.AVAILABLE_MODELS = {}
            st.session_state.MODEL_PATHS = {}
            return
        
        # List all files in the models directory
        st_debug(f"Scanning models directory: {models_dir}")
        model_files = os.listdir(models_dir)
        st_debug(f"Found files: {model_files}")
        
        # Initialize model mappings
        st.session_state.MODEL_PATHS = {}
        available_models = {}
        
        # Process each model file
        for model_file in model_files:
            classifier_type, friendly_name, description, clean_name = parse_model_name(model_file)
            
            if classifier_type:
                model_key = model_file  # Use full filename as key
                model_path = os.path.join(models_dir, clean_name)
                
                st.session_state.MODEL_PATHS[model_key] = model_path
                available_models[model_key] = (friendly_name, description, classifier_type)
                
                st_debug(f"Added model: {model_key}")
                st_debug(f"  Path: {model_path}")
        
        st_debug(f"Total models found: {len(available_models)}")
        st.session_state.AVAILABLE_MODELS = available_models

# # Initialize models at startup
# check_available_models()

def show_debug_logs():
    """Display debug logs in the sidebar."""
    if 'debug_logs' in st.session_state:
        # Show log count
        log_count = len(st.session_state.debug_logs)
        st.sidebar.text(f"Total logs: {log_count}")
        
        # Combine all logs into a single string
        log_text = ""
        for log in reversed(st.session_state.debug_logs[-100:]):
            timestamp = log.get('timestamp', '')
            level = log.get('level', 'info')
            message = log.get('message', '')
            
            # Format log entry
            log_text += f"[{timestamp}] {level.upper()}: {message}\n"
        
        # Display in a text area
        st.sidebar.text_area(
            "",
            value=log_text,
            height=400,
        )

def show_error(title, error, details=None):
    """Display error messages in the main UI."""
    st.error(f"🚫 {title}")
    
    with st.expander("Show Error Details"):
        if details:
            st.warning(details)
        st.error(f"Error Type: {type(error).__name__}")
        st.error(str(error))
        
        # Log error to debug
        if st.session_state.get('debug_enabled'):
            st_debug(f"Error: {title} - {str(error)}", level='error')



def clear_previous_results(keep_types=None):
    """
    Clear previous results and messages before new analysis.
    
    Args:
        keep_types (list, optional): List of analysis types to keep in results
    """
    # Clear UI messages
    if 'ui_messages' in st.session_state:
        st.session_state.ui_messages = []
    
    # Clear analysis results except specified types
    if 'analysis_results' in st.session_state:
        if keep_types:
            st.session_state.analysis_results = {
                k: v for k, v in st.session_state.analysis_results.items() 
                if k in keep_types
            }
        else:
            st.session_state.analysis_results = {}

def toggle_debug_mode():
    """Callback to handle debug mode toggle"""
    st.session_state.debug_enabled = not st.session_state.debug_enabled
    # Don't clear logs when toggling debug mode

def main():
    """
    Main UI function for the audio analysis tool.
    
    Provides interface for:
    - File upload (WAV format)
    - Analysis type selection
    - Model selection
    - Debug mode toggle
    - Results visualization
    
    Settings:
        Debug Mode: Toggle detailed debug information display
        
    Analysis Types:
        - Classification: Various audio classification tasks
        - Regression: Pitch, beat detection, and custom regression
        - Feature Extraction: Short-term and mid-term features
        - Segmentation: Supervised, unsupervised, silence removal
        - Diarization: Speaker identification and segmentation
        - Beat Extraction: Beat detection and analysis
    """
# Initialize models at startup
    check_available_models()

        # Initialize debug mode as enabled by default
    if 'debug_enabled' not in st.session_state:
        st.session_state.debug_enabled = False

    st.set_page_config(
        page_title="Audio Analysis",
        layout="centered",
        initial_sidebar_state="expanded",
        page_icon="🎵"
    )
    

    
    # Initialize debug logs if not present
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    # Initialize models directory only once at startup
    if 'models_initialized' not in st.session_state:
        audio_utils.ensure_models_directory()
        st.session_state.models_initialized = True
    
    # Ensure models directory exists
    audio_utils.ensure_models_directory()
    
    # Initialize session state for analysis results if not present
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Create separate containers
    debug_container = st.sidebar.container()
    main_container = st.container()
    output_container = st.container()
    
    # Debug settings in sidebar
    with debug_container:
        st.header("Debug Settings")
        
        # Debug mode toggle with callback
        st.checkbox(
            "Enable Debug Mode",
            value=st.session_state.debug_enabled,
            key='debug_checkbox',
            on_change=toggle_debug_mode,
            help="Show detailed debug information and error messages"
        )
        
        if st.session_state.debug_enabled:
            if st.button("Clear Debug Logs", key="clear_debug_logs"):
                st.session_state.debug_logs = []
            
            st.markdown("### Live Debug Log")
            show_debug_logs()
    
    # Main UI in its own container
    with main_container:
        st.title("Audio Analysis Tool")
        
        if not st.session_state.AVAILABLE_MODELS:
            st.error("No models available. Please check the models directory.")
            return
            
        # File Upload Section
        uploaded_file = st.file_uploader(
            "Upload an audio file (WAV format only)", 
            type=['wav'],
            help="Only WAV files are supported. If you have an MP3 file, please convert it to WAV first."
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            [Fs, x] = audioBasicIO.read_audio_file(temp_path)
            
            # Main Analysis Options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Classification", "Regression", "Feature Extraction", "Segmentation", "Diarization", "Beat Extraction"]
            )
            
            if analysis_type == "Classification":
                # Only show SVM models as they have proper normalization
                svm_models = {k: v for k, v in st.session_state.AVAILABLE_MODELS.items() if v[2] == 'svm_rbf'}
                
                model_name = st.selectbox(
                    "Select Classification Task",
                    list(svm_models.keys()),
                    format_func=lambda x: svm_models[x][0]  # Show friendly name
                )
                
                # Show model description
                if model_name in st.session_state.AVAILABLE_MODELS:
                    st.info(st.session_state.AVAILABLE_MODELS[model_name][1])
                
                if st.button("Classify"):
                    clear_previous_results()
                    perform_classification(temp_path, model_name)
            
            elif analysis_type == "Regression":
                reg_type = st.selectbox(
                    "Select Regression Task",
                    ["Pitch Detection", "Beat Detection", "Custom Regression"]
                )
                
                if reg_type == "Pitch Detection":
                    st.info("Pitch detection analyzes the fundamental frequency of the audio signal.")
                    if st.button("Detect Pitch"):
                        clear_previous_results()
                        with st.spinner("Analyzing pitch..."):
                            # Extract features
                            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
                            
                            # Get pitch-related features
                            pitch_features = [i for i, name in enumerate(f_names) if 'pitch' in name.lower()]
                            
                            # Create time axis
                            time = np.arange(F.shape[1]) * 0.025  # 25ms steps
                            
                            # Plot pitch over time
                            fig = go.Figure()
                            for idx in pitch_features:
                                fig.add_trace(go.Scatter(
                                    x=time,
                                    y=F[idx, :],
                                    name=f_names[idx],
                                    mode='lines'
                                ))
                            
                            fig.update_layout(
                                title="Pitch Analysis",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Pitch Features",
                                showlegend=True
                            )
                            st.plotly_chart(fig)
                            
                            # Display statistics
                            st.write("### Pitch Statistics")
                            for idx in pitch_features:
                                st.write(f"**{f_names[idx]}**")
                                st.write(f"- Mean: {np.mean(F[idx, :]):.2f}")
                                st.write(f"- Std: {np.std(F[idx, :]):.2f}")
                                st.write(f"- Max: {np.max(F[idx, :]):.2f}")
                                st.write(f"- Min: {np.min(F[idx, :]):.2f}")
                
                elif reg_type == "Beat Detection":
                    st.info("Beat detection identifies rhythmic patterns in the audio.")
                    if st.button("Detect Beats"):
                        clear_previous_results()
                        with st.spinner("Analyzing beats..."):
                            # Extract features
                            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
                            
                            # Detect beats
                            beats = beat_extraction(F, 0.050)[0]
                            
                            # Create time axis
                            time = np.arange(len(beats)) * 0.050  # 50ms steps
                            
                            # Plot beat strength over time
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=time,
                                y=beats,
                                mode='lines',
                                name='Beat Strength'
                            ))
                            
                            # Add markers for strong beats
                            strong_beats = np.where(beats > np.mean(beats) + np.std(beats))[0]
                            fig.add_trace(go.Scatter(
                                x=time[strong_beats],
                                y=beats[strong_beats],
                                mode='markers',
                                name='Strong Beats',
                                marker=dict(size=10, color='red')
                            ))
                            
                            fig.update_layout(
                                title="Beat Detection",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Beat Strength",
                                showlegend=True
                            )
                            st.plotly_chart(fig)
                            
                            # Display beat statistics
                            st.write("### Beat Statistics")
                            st.write(f"- Number of strong beats: {len(strong_beats)}")
                            if len(strong_beats) > 1:
                                beat_intervals = np.diff(time[strong_beats])
                                avg_tempo = 60 / np.mean(beat_intervals)
                                st.write(f"- Estimated tempo: {avg_tempo:.1f} BPM")
                                st.write(f"- Average beat interval: {np.mean(beat_intervals):.3f} seconds")
                
                elif reg_type == "Custom Regression":
                    st.info("Upload your own regression model for custom audio analysis.")
                    model_file = st.file_uploader("Upload regression model", type=['svr', 'pkl'])
                    if model_file and st.button("Analyze"):
                        clear_previous_results()
                        perform_custom_regression(temp_path, model_file)
            
            elif analysis_type == "Feature Extraction":
                feature_type = st.selectbox(
                    "Select Feature Type",
                    ["Short-term Features", "Mid-term Features"]
                )
                
                # Feature extraction parameters
                st.subheader("Parameters")
                win_size = st.slider("Window Size (sec)", 0.02, 1.0, 0.05)
                win_step = st.slider("Window Step (sec)", 0.01, 0.5, 0.025)
                
                if st.button("Extract Features"):
                    if feature_type == "Short-term Features":
                        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 
                                                                        win_size*Fs, 
                                                                        win_step*Fs)
                        display_features(F, f_names)
                    elif feature_type == "Mid-term Features":
                        mt_win_size = st.slider("Mid-term Window Size (sec)", 1.0, 10.0, 1.0)
                        mt_win_step = st.slider("Mid-term Window Step (sec)", 0.1, 5.0, 0.5)
                        [mt_features, st_features, mt_feature_names] = mT(
                            x, Fs, 
                            round(mt_win_size * Fs), 
                            round(mt_win_step * Fs),
                            round(Fs * win_size), 
                            round(Fs * win_size * 0.5)
                        )
                        display_features(mt_features, mt_feature_names)
            
            elif analysis_type == "Segmentation":
                st_debug("Entering segmentation section")
                
                seg_type = st.selectbox(
                    "Select Segmentation Type",
                    ["Unsupervised", "Detect Silence"]
                )
                
                if seg_type == "Unsupervised":
                    n_segments = st.slider("Number of Segments", 2, 10, 4)
                    
                    st_debug("=== UI Segmentation Section ===")
                    st_debug(f"Selected segmentation type: {seg_type}")
                    st_debug(f"Number of segments selected: {n_segments}")
                    
                    # Add a button to trigger segmentation
                    if st.button("Perform Segmentation", key="perform_segmentation"):
                        clear_previous_results()
                        st_debug("Segmentation button clicked")
                        try:
                            # Verify audio data
                            st_debug(f"Audio data shape: {x.shape}")
                            st_debug(f"Sampling rate: {Fs}")
                            
                            # Add progress indicator
                            with st.spinner("Performing segmentation..."):
                                st_debug("Starting segmentation process")
                                
                                # Perform segmentation
                                fig, stats = perform_unsupervised_segmentation(x, Fs, n_segments)
                                
                                # Store results
                                st.session_state.analysis_results["Segmentation"] = {
                                    "chart": fig,
                                    "stats": stats,
                                    "type": "unsupervised"
                                }
                                st_debug("Segmentation completed and results stored")
                                
                                # Remove direct display here - results will be shown in output_container
                        except Exception as e:
                            st.error(f"Segmentation failed: {str(e)}")
                            st_debug(f"Segmentation error: {str(e)}", level='error')
                            st_debug(f"Full error: {traceback.format_exc()}", level='error')
                
                elif seg_type == "Detect Silence":
                    if st.button("Detect Silence"):
                        clear_previous_results()
                        perform_silence_detection(x, Fs)

            elif analysis_type == "Diarization":
                st.info("Speaker diarization identifies and separates different speakers in the audio.")
                n_speakers = st.slider("Expected Number of Speakers", 2, 10, 2)
                
                if st.button("Perform Diarization"):
                    clear_previous_results()
                    with st.spinner("Performing speaker diarization..."):
                        try:
                            # Extract mid-term features
                            mt_size, mt_step, st_win = 2, 0.1, 0.05
                            [mt_feats, st_feats, _] = mT(x, Fs, 
                                                        mt_size * Fs, 
                                                        mt_step * Fs,
                                                        round(Fs * st_win), 
                                                        round(Fs * st_win * 0.5))
                            
                            # Normalize features
                            (mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
                            mt_feats_norm = mt_feats_norm[0].T
                            
                            # Perform clustering
                            k_means = sklearn.cluster.KMeans(n_clusters=n_speakers)
                            k_means.fit(mt_feats_norm.T)
                            cls = k_means.labels_
                            
                            # Convert labels to segments
                            segs, c = labels_to_segments(cls, mt_step)
                            
                            # Create WAV files for each speaker
                            x_clusters = [np.zeros((Fs,)) for i in range(n_speakers)]
                            stats = []
                            speaker_audio = []  # List to store audio data
                            
                            for sp in range(n_speakers):
                                count_cl = 0
                                total_duration = 0
                                for i in range(len(c)):
                                    if c[i] == sp and segs[i, 1] - segs[i, 0] > 2:
                                        count_cl += 1
                                        cur_x = x[int(segs[i, 0] * Fs): int(segs[i, 1] * Fs)]
                                        x_clusters[sp] = np.append(x_clusters[sp], cur_x)
                                        x_clusters[sp] = np.append(x_clusters[sp], np.zeros((Fs,)))
                                        total_duration += segs[i, 1] - segs[i, 0]
                                
                                stats.append({
                                    'speaker': sp,
                                    'segments': count_cl,
                                    'duration': total_duration
                                })
                                
                                # Store audio data instead of creating temporary files
                                speaker_audio.append({
                                    'data': np.int16(x_clusters[sp]),
                                    'fs': Fs
                                })
                            
                            # Store results including audio data
                            st.session_state.analysis_results["Diarization"] = {
                                "segments": segs,
                                "classes": c,
                                "stats": stats,
                                "mt_step": mt_step,
                                "speaker_audio": speaker_audio
                            }
                            
                        except Exception as e:
                            st.error("Diarization error occurred")
                            with st.expander("Error Details"):
                                st.error(str(e))
                                st.write("Stack Trace:")
                                st.code(traceback.format_exc())
                    
            elif analysis_type == "Beat Extraction":
                st.info("Beat extraction analyzes rhythmic patterns in the audio.")
                
                if st.button("Extract Beats"):
                    clear_previous_results()
                    try:
                        # Extract features
                        window_size = 0.050  # 50ms windows
                        F, f_names = ShortTermFeatures.feature_extraction(
                            x, Fs, 
                            int(window_size * Fs), 
                            int(window_size * Fs)
                        )
                        
                        # Extract beats
                        energy, beats, tempo = beat_extraction(F, window_size)
                        time = np.arange(len(energy)) * window_size
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Plot energy
                        fig.add_trace(go.Scatter(
                            x=time,
                            y=energy,
                            mode='lines',
                            name='Energy'
                        ))
                        
                        # Add beat markers
                        if len(beats) > 0:
                            fig.add_trace(go.Scatter(
                                x=time[beats],
                                y=energy[beats],
                                mode='markers',
                                name='Beats',
                                marker=dict(size=10, color='red')
                            ))
                        
                        fig.update_layout(
                            title="Beat Detection",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Energy",
                            showlegend=True
                        )
                        
                        # Store results
                        st.session_state.analysis_results["Beat Extraction"] = {
                            "chart": fig,
                            "tempo": tempo,
                            "beats": beats,
                            "time": time,
                            "energy": energy
                        }
                        
                    except Exception as e:
                        st.error(f"Beat extraction failed: {str(e)}")
                        st_debug(f"Beat extraction error: {str(e)}", level='error')
            
            # Cleanup
            os.unlink(temp_path)

        st_debug("UI initialized")
        audio_utils.display_ui_messages()

    # Display messages and results in the same container
    with output_container:
        # Display UI messages
        audio_utils.display_ui_messages()
        
        # Display analysis results
        if st.session_state.analysis_results:
            for analysis_type, results in st.session_state.analysis_results.items():
                if analysis_type == "Classification":
                    # Removed redundant header
                    st.success(f"""
                        Audio classified as: **{results['predicted_class']}**
                    """)
                    
                    # Show confidence scores
                    st.markdown("### Confidence Scores")
                    for class_name, prob in zip(results['class_names'], results['probabilities']):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{class_name}:")
                            st.progress(float(prob))
                        with col2:
                            st.write(f"{prob*100:.1f}%")
                        st.write("---")
                    
                    # Plot results
                    fig = audio_utils.plot_classification_results(
                        results['class_names'], 
                        results['probabilities']
                    )
                    if fig:
                        st.plotly_chart(fig)
                elif analysis_type == "Beat Extraction":
                    st.plotly_chart(results["chart"])
                    if results["tempo"] > 0:
                        st.write(f"Estimated tempo: {results['tempo']:.1f} BPM")
                    st.write(f"Number of beats detected: {len(results['beats'])}")
                elif analysis_type == "Segmentation":
                    st.plotly_chart(results["chart"], key=f"stored_{analysis_type}_chart")
                    if results.get("stats"):
                        st.write("### Cluster Statistics")
                        for stat in results["stats"]:
                            st.write(f"**Cluster {stat['cluster']}**")
                            st.write(f"- Duration: {stat['duration']:.2f} seconds")
                            st.write(f"- Percentage: {stat['percentage']:.1f}%")
                            st.write(f"- Number of segments: {stat['n_segments']}")
                            st.write(f"- Average segment duration: {stat['avg_segment_duration']:.2f} seconds")
                elif analysis_type == "Silence Detection":
                    st.subheader("Silence Detection Results")
                    stats = results["stats"]
                    
                    # Display statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Duration", f"{stats['total_duration']:.2f}s")
                        st.metric("Silent Duration", f"{stats['silent_duration']:.2f}s")
                    with col2:
                        st.metric("Non-Silent Duration", f"{stats['non_silent_duration']:.2f}s")
                        st.metric("Silent Percentage", f"{stats['percent_silent']:.1f}%")
                    
                    # Display segments timeline
                    segments = results["segments"]
                    fig = go.Figure()
                    for i, seg in enumerate(segments):
                        fig.add_trace(go.Scatter(
                            x=[seg[0], seg[1]],
                            y=[1, 1],
                            mode='lines',
                            name=f'Segment {i+1}',
                            showlegend=False
                        ))
                    fig.update_layout(
                        title="Non-Silent Segments Timeline",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Audio",
                        yaxis_range=[0, 2],
                        yaxis_showticklabels=False
                    )
                    st.plotly_chart(fig)
                elif analysis_type == "Diarization":
                    st.subheader("Speaker Diarization Results")
                    
                    # Display statistics
                    for stat in results["stats"]:
                        st.write(f"**Speaker {stat['speaker'] + 1}**")
                        st.write(f"- Number of segments: {stat['segments']}")
                        st.write(f"- Total duration: {stat['duration']:.2f} seconds")
                    
                    # Create download buttons for each speaker
                    for i, audio_data in enumerate(results["speaker_audio"]):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            wavfile.write(tmp_file.name, audio_data['fs'], audio_data['data'])
                            with open(tmp_file.name, 'rb') as f:
                                st.download_button(
                                    f"Download Speaker {i+1} Audio",
                                    f,
                                    file_name=f"speaker_{i+1}.wav",
                                    mime="audio/wav",
                                    key=f"speaker_{i}"
                                )
                            os.unlink(tmp_file.name)

def display_features(F, f_names):
    st.subheader("Extracted Features")
    
    # Create feature plots
    for i, name in enumerate(f_names):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(F[i, :])
        plt.title(name)
        st.pyplot(fig)
        plt.close()

def check_model_path(model_path):
    """Verify model path and return status"""
    st_debug(f"Checking model path: {model_path}")
    st_debug(f"Absolute path: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st_debug(f"Model file does not exist: {model_path}")
        
        # Check parent directory
        parent_dir = os.path.dirname(model_path)
        if os.path.exists(parent_dir):
            st_debug(f"Parent directory exists: {parent_dir}")
            st_debug(f"Contents of parent directory: {os.listdir(parent_dir)}")
        else:
            st_debug(f"Parent directory does not exist: {parent_dir}")
        
        st_debug(f"Please ensure models are in the correct directory structure:")
        st_debug("""
        models/
        ├── svm_rbf_sm (Speech/Music Classification)
        ├── svm_rbf_speaker (Speaker Recognition)
        ├── svm_rbf_movie8 (Movie Genre)
        ├── svm_rbf_speaker-gender (Speaker Gender)
        ├── svm_rbf_music-genre6 (Music Genre)
        └── svm_rbf_4class (4-Class Audio)
        """)
        return False
        
    st_debug(f"Model file found at: {model_path}")
    return True

def perform_classification(audio_path, task_type):
    """
    Perform audio classification based on task type.
    """
    try:
        # Validate audio file
        valid, Fs, x = audio_utils.read_audio_file(audio_path)
        if not valid:
            return

        # Check model path
        if task_type not in st.session_state.MODEL_PATHS:
            audio_utils.st_message(f"Classification task '{task_type}' not found", level='error')
            return
            
        model_path = st.session_state.MODEL_PATHS[task_type]
        if not audio_utils.check_model_path(model_path):
            return
            
        # Create progress bar
        progress_text = "Classification in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            # Perform classification steps
            my_bar.progress(25, text="Extracting audio features...")
            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
            
            my_bar.progress(50, text="Loading classification model...")
            result = aT.file_classification(audio_path, model_path, "svm_rbf")
            
            my_bar.progress(100, text="Classification complete!")
            my_bar.empty()
            
            # Store and display results
            if isinstance(result, tuple) and len(result) == 3:
                class_id, probabilities, class_names = result
                predicted_class = class_names[int(class_id)]
                
                # Store in session state
                st.session_state.analysis_results["Classification"] = {
                    'predicted_class': predicted_class,
                    'class_names': class_names,
                    'probabilities': probabilities,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
                # audio_utils.st_message("Classification completed successfully!", level='success')
                # audio_utils.st_message(f"Audio classified as: {predicted_class}", level='info')
                
            else:
                audio_utils.st_message("Classification failed: Invalid model output", level='error')
                
        except Exception as e:
            my_bar.empty()
            raise e
            
    except Exception as e:
        audio_utils.st_message("Classification error occurred", level='error')
        with st.expander("Error Details"):
            audio_utils.st_message(str(e), level='error')
            audio_utils.st_message("Stack Trace:", level='error')
            audio_utils.st_message(traceback.format_exc(), level='error')

def perform_unsupervised_segmentation(x, Fs, n_segments):
    """Perform unsupervised audio segmentation using clustering."""
    try:
        audio_utils.st_debug(f"Audio signal shape: {x.shape}")
        audio_utils.st_debug(f"Sampling rate: {Fs} Hz")
        audio_utils.st_debug(f"Signal duration: {len(x)/Fs:.2f} seconds")
        
        # Extract mid-term features
        mt_size, mt_step = 1.0, 0.1
        st_win = 0.05
        
        audio_utils.st_debug("Extracting mid-term features...")
        [mt_feats, st_feats, _] = mT(
            x, Fs, 
            round(mt_size * Fs), 
            round(mt_step * Fs),
            round(Fs * st_win), 
            round(Fs * st_win * 0.5)
        )
        
        audio_utils.st_debug(f"Mid-term features shape: {mt_feats.shape}")
        
        # Normalize features
        (mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
        mt_feats_norm = mt_feats_norm[0].T
        
        # Calculate actual number of segments
        n_samples = mt_feats_norm.shape[1]
        actual_n_segments = min(n_segments, n_samples)
        
        # Perform clustering with fixed random seed
        audio_utils.st_debug(f"Performing clustering with {actual_n_segments} segments...")
        k_means = sklearn.cluster.KMeans(
            n_clusters=actual_n_segments,
            random_state=42
        )
        k_means.fit(mt_feats_norm.T)
        cls = k_means.labels_
        
        audio_utils.st_debug(f"Clustering complete. Labels shape: {cls.shape}")
        
        # Convert labels to segments
        segs = []
        cur_segment_start = 0
        cur_label = cls[0]
        
        # Create segments based on label changes
        for i in range(1, len(cls)):
            if cls[i] != cur_label:
                segs.append([cur_segment_start * mt_step, i * mt_step, cur_label])
                cur_segment_start = i
                cur_label = cls[i]
        
        # Add final segment
        segs.append([cur_segment_start * mt_step, len(cls) * mt_step, cur_label])
        segs = np.array(segs)
        
        audio_utils.st_debug(f"Created {len(segs)} segments")
        
        # Create visualization and calculate statistics
        fig = create_segmentation_visualization(segs, cls, actual_n_segments, len(x)/Fs)
        stats = calculate_segment_statistics(segs, cls, actual_n_segments, len(x)/Fs)
        
        return fig, stats
        
    except Exception as e:
        audio_utils.st_debug(f"Segmentation error: {str(e)}", level='error')
        audio_utils.st_debug(f"Traceback: {traceback.format_exc()}", level='error')
        raise

def perform_silence_detection(x, Fs):
    """Detect and analyze silent segments in audio."""
    try:
        
        # Detect silence
        segments = aS.silence_removal(x, Fs, 0.050, 0.050, smooth_window=1.0)
        
        if not segments:
            st.warning("No non-silent segments found in the audio")
            return
        
        # Calculate statistics
        total_duration = len(x) / Fs
        non_silent_duration = sum(seg[1] - seg[0] for seg in segments)
        silent_duration = total_duration - non_silent_duration
        
        # Store results
        st.session_state.analysis_results["Silence Detection"] = {
            "segments": segments,
            "stats": {
                "total_duration": total_duration,
                "non_silent_duration": non_silent_duration,
                "silent_duration": silent_duration,
                "percent_silent": (silent_duration / total_duration) * 100
            }
        }
        
    except Exception as e:
        audio_utils.st_message("Silence detection failed", level='error')
        audio_utils.st_message(str(e), level='error')
        st_debug(f"Silence detection error: {str(e)}", level='error')

def perform_regression(audio_path, reg_type):
    """
    Perform audio regression analysis.
    
    Args:
        audio_path (str): Path to the audio file
        reg_type (str): Type of regression analysis to perform
            Options: "Pitch Detection", "Beat Detection", "Custom Regression"
            
    Displays:
        - Regression results visualization
        - Statistical analysis
        - Feature plots
    """
    try:
        if reg_type == "Pitch Detection":
            [Fs, x] = audioBasicIO.read_audio_file(audio_path)
            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
            
            # Plot pitch-related features
            pitch_features = ['spectral_centroid', 'spectral_rolloff']
            for feature in pitch_features:
                if feature in f_names:
                    idx = f_names.index(feature)
                    fig = plt.figure(figsize=(10, 4))
                    plt.plot(F[idx, :])
                    plt.title(feature)
                    st.pyplot(fig)
                    plt.close()
                    
        elif reg_type == "Beat Detection":
            [Fs, x] = audioBasicIO.read_audio_file(audio_path)
            F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.050*Fs)
            # Detect beats
            beats = beat_extraction(F, 0.050)[0]
            
            # Create beat visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[i * 0.050 for i in range(len(beats))],
                y=beats,
                mode='lines',
                name='Beat Strength'
            ))
            
            fig.update_layout(
                title="Beat Detection",
                xaxis_title="Time (seconds)",
                yaxis_title="Beat Strength",
                showlegend=True
            )
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Regression error: {str(e)}")

def perform_custom_regression(audio_path, model_file):
    """
    Perform regression using a custom uploaded model.
    
    Args:
        audio_path (str): Path to the audio file
        model_file (UploadedFile): Custom regression model file (SVR or pickle format)
        
    Displays:
        - Regression predictions
        - Model performance metrics
    """
    try:
        # Save model file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(model_file.getvalue())
            model_path = tmp_file.name
            
        # Extract features and perform regression
        [Fs, x] = audioBasicIO.read_audio_file(audio_path)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
        
        # Load model and predict
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        predictions = model.predict(F.T)
        
        # Plot predictions
        fig = plt.figure(figsize=(10, 4))
        plt.plot(predictions)
        plt.title("Regression Predictions")
        st.pyplot(fig)
        plt.close()
        
        # Cleanup
        os.unlink(model_path)
        
    except Exception as e:
        st.error(f"Custom regression error: {str(e)}")

def beat_extraction(features, window_size):
    """Extract beats from audio features."""
    try:
        audio_utils.st_debug("Starting beat extraction...")
        
        # Calculate energy
        energy = features[1, :]  # Energy feature
        
        # Find peaks in energy using peak detection
        peaks, _ = find_peaks(energy, distance=int(0.5/window_size))
        
        # Calculate tempo
        if len(peaks) > 1:
            # Average time between peaks
            peak_times = peaks * window_size
            intervals = np.diff(peak_times)
            mean_interval = np.mean(intervals)
            tempo = 60.0 / mean_interval
        else:
            tempo = 0
            st.warning("Not enough beats detected to calculate tempo")  # Use st.warning directly
            
        audio_utils.st_debug(f"Found {len(peaks)} beats, tempo: {tempo:.1f} BPM")
        return energy, peaks, tempo
        
    except Exception as e:
        st.error("Beat extraction failed")
        audio_utils.st_debug(f"Beat extraction error: {str(e)}", level='error')
        raise

def normalize_features(features):
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features (list): List of feature matrices
        
    Returns:
        tuple: (normalized_features, mean, std)
    """
    temp_features = np.array([])
    
    for f in features:
        temp_features = np.vstack((temp_features, f)) if temp_features.size else f
    
    mean = np.mean(temp_features, axis=0)
    std = np.std(temp_features, axis=0)
    
    normalized_features = []
    for f in features:
        f_norm = (f - mean) / std
        normalized_features.append(f_norm)
    
    return normalized_features, mean, std

def create_segmentation_visualization(segments, classes, n_clusters, duration):
    """Create plotly visualization of audio segments."""
    fig = go.Figure()
    
    for i in range(n_clusters):
        # Find segments for this cluster
        cluster_segs = [seg for seg in segments if seg[2] == i]  # Use list comprehension instead of boolean indexing
        
        # Add segments to plot
        for seg in cluster_segs:
            fig.add_trace(go.Scatter(
                x=[seg[0], seg[1]],
                y=[i, i],
                mode='lines',
                name=f'Cluster {i}',
                showlegend=False
            ))
    
    fig.update_layout(
        title="Audio Segmentation",
        xaxis_title="Time (seconds)",
        yaxis_title="Cluster",
        yaxis_range=[-0.5, n_clusters-0.5]
    )
    
    return fig

def calculate_segment_statistics(segments, classes, n_clusters, duration):
    """Calculate statistics for each segment cluster."""
    stats = []
    for i in range(n_clusters):
        cluster_segs = [seg for seg in segments if seg[2] == i]  # Use list comprehension instead of boolean indexing
        total_duration = sum(seg[1] - seg[0] for seg in cluster_segs)
        stats.append({
            'cluster': i,
            'duration': total_duration,
            'percentage': (total_duration / duration) * 100,
            'n_segments': len(cluster_segs),
            'avg_segment_duration': total_duration / len(cluster_segs) if len(cluster_segs) > 0 else 0
        })
    return stats

if __name__ == "__main__":
    main() 