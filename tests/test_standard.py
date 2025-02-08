"""Standard functionality tests previously run through shell scripts."""

import pytest
import os
import shutil
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioAnalysis

@pytest.fixture(scope="session")
def test_data_path():
    """Provide path to test data directory."""
    base_path = os.path.dirname(os.path.dirname(__file__))  # Get project root directory
    print(f"Base path: {base_path}")
    return os.path.join(base_path, "pyAudioAnalysis/data")  # Return path to data directory

@pytest.fixture(scope="session", autouse=True)
def setup_test_data(test_data_path):
    """Automatically setup test data before running tests."""
    print(f"Test data path: {test_data_path}")
    # Create test data directories if they don't exist
    os.makedirs(os.path.join(test_data_path, "music"), exist_ok=True)
    os.makedirs(os.path.join(test_data_path, "speech"), exist_ok=True)
    os.makedirs(os.path.join(test_data_path, "models"), exist_ok=True)
    
    return test_data_path

def check_test_file(test_data_path, filename):
    """Check if a specific test file exists."""
    file_path = os.path.join(test_data_path, filename)
    if not os.path.exists(file_path):
        pytest.skip(f"Required test file not found: {filename}")
    return file_path

@pytest.fixture(scope="session")
def models_path(test_data_path):
    """Provide path to models directory."""
    return os.path.join(test_data_path, "models")

# Tests from cmd_test_00.sh
class TestBasicFunctionality:
    """Tests for basic audio analysis functionality."""
    
    def test_feature_extraction(self, test_data_path):
        """Test short-term feature extraction (cmd_test_00)."""
        audio_file = check_test_file(test_data_path, "doremi.wav")
        [Fs, x] = audioBasicIO.read_audio_file(audio_file)
        
        # Test short-term feature extraction (matches cmd_test_00.sh)
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
        
        # Just verify we get valid output
        assert F is not None
        assert f_names is not None
        assert F.shape[0] > 0
        assert len(f_names) > 0

    def test_cmd_00_spectrogram(self, test_data_path):
        """Test command from cmd_test_00.sh:
        python3 audioAnalysis.py fileSpectrogram -i data/doremi.wav
        """
        import matplotlib
        matplotlib.use('Agg')  # Set backend to non-interactive
        import matplotlib.pyplot as plt
        from pyAudioAnalysis.audioAnalysis import fileSpectrogramWrapper
        
        # Check if test file exists
        audio_file = check_test_file(test_data_path, "doremi.wav")
        
        try:
            # Get current number of figures
            n_figs_before = len(plt.get_fignums())
            
            # Call the wrapper function but don't show the plot
            with plt.ion():  # Turn on interactive mode to prevent show() from blocking
                fileSpectrogramWrapper(audio_file)
            
            # Verify a new figure was created
            n_figs_after = len(plt.get_fignums())
            assert n_figs_after > n_figs_before, "No spectrogram figure was created"
            
            # Clean up
            plt.close('all')
                
        except Exception as e:
            plt.close('all')  # Ensure cleanup even if test fails
            pytest.fail(f"Spectrogram generation failed with error: {str(e)}")

    def test_cmd_01_chromagram(self, test_data_path):
        """Test command from cmd_test_01.sh:
        python3 audioAnalysis.py fileChromagram -i data/doremi.wav
        """
        import matplotlib
        matplotlib.use('Agg')  # Set backend to non-interactive
        import matplotlib.pyplot as plt
        from pyAudioAnalysis.audioAnalysis import fileChromagramWrapper
        
        # Check if test file exists
        audio_file = check_test_file(test_data_path, "doremi.wav")
        
        try:
            # Get current number of figures
            n_figs_before = len(plt.get_fignums())
            
            # Call the wrapper function but don't show the plot
            with plt.ion():  # Turn on interactive mode to prevent show() from blocking
                fileChromagramWrapper(audio_file)
            
            # Verify a new figure was created
            n_figs_after = len(plt.get_fignums())
            assert n_figs_after > n_figs_before, "No chromagram figure was created"
            
            # Clean up
            plt.close('all')
                
        except Exception as e:
            plt.close('all')  # Ensure cleanup even if test fails
            pytest.fail(f"Chromagram generation failed with error: {str(e)}")

    def test_cmd_05_hmm_segmentation(self, test_data_path):
        """Test command from cmd_test_05.sh:
        python3 audioAnalysis.py segmentClassifyFileHMM -i data/scottish.wav --hmm data/hmmRadioSM
        """
        import matplotlib
        matplotlib.use('Agg')  # Set backend to non-interactive
        import matplotlib.pyplot as plt
        from pyAudioAnalysis.audioAnalysis import segmentclassifyFileWrapperHMM
        
        # Check if required files exist
        audio_file = check_test_file(test_data_path, "scottish.wav")
        hmm_model = check_test_file(test_data_path, "hmmRadioSM")
        
        try:
            # Get current number of figures
            n_figs_before = len(plt.get_fignums())
            
            # Call the wrapper function but don't show the plot
            with plt.ion():  # Turn on interactive mode to prevent show() from blocking
                segmentclassifyFileWrapperHMM(audio_file, hmm_model)
            
            # Verify a new figure was created
            n_figs_after = len(plt.get_fignums())
            assert n_figs_after > n_figs_before, "No HMM segmentation figure was created"
            
            # Clean up
            plt.close('all')
                
        except Exception as e:
            plt.close('all')  # Ensure cleanup even if test fails
            pytest.fail(f"HMM segmentation failed with error: {str(e)}")
