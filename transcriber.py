import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, model_path, sample_rate=16000, record_seconds=10):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.compute_type = "int8"
        self.model = WhisperModel(model_path, device="cpu", compute_type=self.compute_type)

    def record_audio(self):
        print("Recording for {} seconds...".format(self.record_seconds))
        recording = sd.rec(int(self.record_seconds * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float64')
        
        for remaining in range(self.record_seconds, 0, -1):
            print(f"Recording... {remaining} seconds remaining", end="\r")
            time.sleep(1)
        
        sd.wait()  # Wait until the recording is finished
        print("Recording complete.")
        return recording

    def save_temp_audio(self, recording):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        write(temp_file.name, self.sample_rate, recording)
        return temp_file.name

    def transcribe_audio(self, file_path):
        start_time = time.time()  # Start time
        segments, info = self.model.transcribe(file_path, beam_size=5)
        end_time = time.time()  # End time
        
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        full_transcription = ""
        for segment in segments:
            print(segment.text)
            full_transcription += segment.text + " "
        
        transcription_time = end_time - start_time  # Calculate transcription time
        print("Time taken to transcribe: {:.2f} seconds".format(transcription_time))
        
        os.remove(file_path)
        return full_transcription

    def run(self):
        while True:
            recording = self.record_audio()
            file_path = self.save_temp_audio(recording)
            transcription = self.transcribe_audio(file_path)
            print("\nTranscription: ", transcription)
            print("\nRecording for another {} seconds...".format(self.record_seconds))

if __name__ == "__main__":
    model_path = "/home/mani/workspace/live_transcription/faster-whisper-large-v3"
    transcriber = WhisperTranscriber(model_path=model_path)
    transcriber.run()
