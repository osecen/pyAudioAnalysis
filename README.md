# <img src="icon.png" align="left" height="130"/> A Python library for audio feature extraction, classification, segmentation and applications

*This is general info. Click [here](https://github.com/tyiannak/pyAudioAnalysis/wiki) for the complete wiki and [here](https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y) for a more generic intro to audio data handling*

## News
 * [2022-01-01] If you are not interested in training audio models from your own data, you can check the [Deep Audio API](https://labs-repos.iit.demokritos.gr/MagCIL/deep_audio_api.html), were you can directly send audio data and receive predictions with regards to the respective audio content (speech vs silence, musical genre, speaker gender, etc). 
 * [2021-08-06] [deep-audio-features](https://github.com/tyiannak/deep_audio_features) deep audio classification and feature extraction using CNNs and Pytorch 
 * Check out [paura](https://github.com/tyiannak/paura) a Python script for realtime recording and analysis of audio data

## General
pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. Through pyAudioAnalysis you can:
 * Extract audio *features* and representations (e.g. mfccs, spectrogram, chromagram)
 * *Train*, parameter tune and *evaluate* classifiers of audio segments
 * *Classify* unknown sounds
 * *Detect* audio events and exclude silence periods from long recordings
 * Perform *supervised segmentation* (joint segmentation - classification)
 * Perform *unsupervised segmentation* (e.g. speaker diarization) and extract audio *thumbnails*
 * Train and use *audio regression* models (example application: emotion recognition)
 * Apply dimensionality reduction to *visualize* audio data and content similarities

## An audio classification example
> More examples and detailed tutorials can be found [at the wiki](https://github.com/tyiannak/pyAudioAnalysis/wiki)

pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. Eg, this code first trains an audio segment classifier, given a set of WAV files stored in folders (each folder representing a different class) and then the trained classifier is used to classify an unknown audio WAV file

```python
from pyAudioAnalysis import audioTrainTest as aT
aT.extract_features_and_train(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.file_classification("data/doremi.wav", "svmSMtemp","svm")
```

>Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])

In addition, command-line support is provided for all functionalities. E.g. the following command extracts the spectrogram of an audio signal stored in a WAV file: `python audioAnalysis.py fileSpectrogram -i data/doremi.wav`

## Further reading

Apart from this README file, to bettern understand how to use this library one should read the following:
  * [Audio Handling Basics: Process Audio Files In Command-Line or Python](https://hackernoon.com/audio-handling-basics-how-to-process-audio-files-using-python-cli-jo283u3y), if you want to learn how to handle audio files from command line, and some basic programming on audio signal processing. Start with that if you don't know anything about audio. 
  * [Intro to Audio Analysis: Recognizing Sounds Using Machine Learning](https://hackernoon.com/intro-to-audio-analysis-recognizing-sounds-using-machine-learning-qy2r3ufl) This goes a bit deeper than the previous article, by providing a complete intro to theory and practice of audio feature extraction, classification and segmentation (includes many Python examples).
 * [The library's wiki](https://github.com/tyiannak/pyAudioAnalysis/wiki)
 * [How to Use Machine Learning to Color Your Lighting Based on Music Mood](https://hackernoon.com/how-to-use-machine-learning-to-color-your-lighting-based-on-music-mood-bi163u8l). An interesting use-case of using this lib to train a real-time music mood estimator.
  * A more general and theoretic description of the adopted methods (along with several experiments on particular use-cases) is presented [in this publication](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610). *Please use the following citation when citing pyAudioAnalysis in your research work*:
```python
@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}
```

For Matlab-related audio analysis material check  [this book](http://www.amazon.com/Introduction-Audio-Analysis-MATLAB%C2%AE-Approach/dp/0080993885).

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Principal Researcher of Multimodal Machine Learning at the [Multimedia Analysis Group of the Computational Intelligence Lab (MagCIL)](https://labs-repos.iit.demokritos.gr/MagCIL/index.html) of the Institute of Informatics and Telecommunications, of the National Center for Scientific Research "Demokritos"

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/tyiannak/pyAudioAnalysis.git
cd pyAudioAnalysis
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies and package:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
## Using the UI Interface

pyAudioAnalysis now includes a web-based UI interface for easy access to all functionality:

### Starting the UI
1. Ensure you're in your virtual environment:
```bash
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. Launch the UI:
```bash
PYTHONPATH=$(pwd) streamlit run pyAudioAnalysis/audioUI.py
```

3. Your default web browser will automatically open to the UI (typically http://localhost:8501)

### UI Features
The interface provides easy access to:
* **Audio Classification**: Upload and classify audio files using pre-trained models
* **Feature Extraction**: Visualize audio features like MFCCs, spectrograms
* **Beat Extraction**: Analyze rhythm and tempo
* **Segmentation**: Perform audio segmentation tasks
* **Regression**: Train and use regression models

### Usage Tips
* Audio files must be in WAV format
* For MP3 files, convert to WAV first using FFmpeg:
```bash
ffmpeg -i input.mp3 output.wav
```
* Models should be trained first using the command line interface before using them in the UI
* Large audio files may take longer to process - consider splitting them into smaller segments

## Running Tests

### Basic Testing Commands
```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage report
python -m pytest --cov=pyAudioAnalysis tests/

# Run specific test file
python -m pytest tests/test_standard.py

# Run tests verbosely
python -m pytest -v tests/

# Run tests matching specific pattern
python -m pytest -k "test_feature" tests/
```

### Test Organization
Test files are organized by functionality:
- `test_standard.py`: Core functionality tests (previously in shell scripts)
- `test_audio_utils.py`: Utility function tests
- `test_ui.py`: Streamlit UI tests

### Coverage Reports
```bash
# Generate HTML coverage report
python -m pytest --cov=pyAudioAnalysis --cov-report=html tests/

# Generate both coverage and test reports
python -m pytest tests/ --cov=pyAudioAnalysis --cov-report=html --html=tests/test-report.html
```

The HTML coverage report will be available in the `htmlcov` directory, and the test report will be in `tests/test-report.html`.

### Continuous Integration
The project uses GitHub Actions for continuous integration, running:
- All tests with coverage reporting
- Code style checks
- System dependency verification
- Multiple Python version testing

Test reports and coverage information are automatically uploaded as artifacts and to Codecov.

### Troubleshooting Tests
If you encounter any issues:

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Verify system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libavcodec-extra

# MacOS
brew install ffmpeg
```

3. Check that you're in the correct directory and virtual environment is activated

4. If you get audio-related errors, ensure your system's audio drivers are properly configured

## Command Line Usage
For those who prefer command line usage, all features are still available through the CLI:

```python
from pyAudioAnalysis import audioTrainTest as aT
# Train a classifier
aT.extract_features_and_train(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
# Classify a file
aT.file_classification("data/doremi.wav", "svmSMtemp","svm")
```

## Model Organization for UI

The UI looks for pre-trained models in the `data/models` directory. To use the classification features in the UI:

1. Create a models directory structure:
```bash
pyAudioAnalysis/
├── data/
│   └── models/
│       ├── svm_rbf_sm.svm         # Speech/Music classifier
│       ├── svm_rbf_4class.svm     # 4-class audio classifier
│       ├── knn_speaker_10.knn     # 10-speaker recognition
│       ├── svm_rbf_movie8.svm     # Movie genre classifier
│       └── svm_rbf_musical_genre_6.svm  # Music genre classifier
```

2. Model Naming Convention:
- Format: `{classifier_type}_{task_name}.{ext}`
- Classifier types: `svm_rbf` or `knn`
- Extensions: `.svm` for SVM models, `.knn` for KNN models

3. Default Model Search Paths:
```
./data/models
../data/models
{package_directory}/data/models
```

If you're using pre-trained models, place them in one of these locations. The UI will automatically detect and list available models in the classification section.

Note: You can train your own models using the command line interface and place them in the models directory for use in the UI.