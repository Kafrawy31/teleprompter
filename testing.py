import tkinter as tk
from tkinter import scrolledtext
import pyaudio
import whisper
import numpy as np
import wave
import tempfile
import os
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Transcription Parameters
sample_rate = 44000
chunk_duration = 2 
overlap_duration = 0.5
chunk_size = int(sample_rate * chunk_duration) 
overlap_size = int(sample_rate * overlap_duration) 
channels = 1

model = whisper.load_model("base.en")

def transcribe_stream(gui_text_widget):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    previous_overlap = np.array([])  # Buffer to store the overlap from the previous chunk

    try:
        while True:
            audio_data = stream.read(chunk_size)
            current_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            combined_chunk = np.concatenate((previous_overlap, current_chunk))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_file_name = temp_audio_file.name
                with wave.open(temp_file_name, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(sample_rate)
                    wf.writeframes((combined_chunk * 32768).astype(np.int16).tobytes())

            # Transcribe the temporary audio file
            result = model.transcribe(temp_file_name, fp16=False)

            os.remove(temp_file_name)

            gui_text_widget.insert(tk.END, result["text"] + " ")
            gui_text_widget.see(tk.END)

            previous_overlap = current_chunk[-overlap_size:]
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Tkinter GUI Setup
def start_gui():
    root = tk.Tk()
    root.title("Real-Time Transcriber")

    transcription_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Arial", 14))
    transcription_text.pack(padx=10, pady=10)
    
    transcription_thread = threading.Thread(target=transcribe_stream, args=(transcription_text,))
    transcription_thread.daemon = True
    transcription_thread.start()

    root.mainloop()

if __name__ == "__main__":
    start_gui()
