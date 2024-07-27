
# Whisper Transcriber

Whisper Transcriber is a Python script that records audio, saves it temporarily, and transcribes the audio using the Faster Whisper model. This project uses the `sounddevice` library for recording audio and the `faster-whisper` library for transcription.

## Features

- Record audio from your microphone for a specified duration.
- Save the recorded audio as a temporary WAV file.
- Transcribe the recorded audio using the Faster Whisper model.
- Display the transcription and the time taken to transcribe.

## Requirements

- Python 3.7 or higher
- `numpy`
- `sounddevice`
- `scipy`
- `faster-whisper`
- `tempfile`
- `os`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/whisper-transcriber.git
    cd whisper-transcriber
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install numpy sounddevice scipy faster-whisper
    ```

4. **Download the Faster Whisper model:**

    ```bash
    git clone https://huggingface.co/Systran/faster-whisper-large-v3
    ```

## Usage

1. **Run the transcriber script:**

    ```bash
    python transcriber.py
    ```

    The script will:
    - Record audio for the specified duration (default is 10 seconds).
    - Display a countdown timer during the recording.
    - Save the recorded audio as a temporary WAV file.
    - Transcribe the audio and display the transcription along with the time taken.

## Using CUDA for Transcription

If you have a CUDA-compatible GPU and want to use it for transcription, update the `WhisperTranscriber` class to use the CUDA device. Replace the initialization line for the Whisper model in the script as follows:

```python
self.model = WhisperModel(model_path, device="cuda", compute_type=float64)
