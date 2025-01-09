import tkinter as tk
import pyaudio
import whisper
import numpy as np
import wave
import tempfile
import os
import threading
import warnings
import string
from fuzzywuzzy import fuzz
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# Transcription Parameters
sample_rate = 44000
chunk_duration = 2.5 
overlap_duration = 0.3
chunk_size = int(sample_rate * chunk_duration) 
overlap_size = int(sample_rate * overlap_duration) 
channels = 1

model = whisper.load_model("base.en")

# Global variable for scrolling index
current_line_index = 0
transcription_log = []  # Store the transcription for output

# Function to split text into 12-word lines
def split_script_into_lines(script):
    words = script.split()
    return [" ".join(words[i:i + 12]) for i in range(0, len(words), 12)]

def transcribe_stream(script_display, entered_script, root):
    global current_line_index, transcription_log
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    previous_overlap = np.array([])  # Buffer to store the overlap from the previous chunk
    script_lines = split_script_into_lines(entered_script)  # Split the script into 12-word lines
    last_transcribed_text = ""  # Store the last processed transcription

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

            # Filter out "um", "okay", "alright", and "all right"
            filtered_text = []
            for word in result["text"].split():
                normalized_word = word.strip(string.punctuation).lower()
                if normalized_word in ["um", "okay", "alright", "all right"]:
                    print(f"Filtered word detected: {word}")  # Log to console
                else:
                    filtered_text.append(word)

            if filtered_text and current_line_index < len(script_lines):
                current_transcription = " ".join(filtered_text)

                # Compare with the current script line
                similarity = fuzz.partial_ratio(current_transcription.lower(), script_lines[current_line_index].lower())
                print(f"Fuzzy similarity for line {current_line_index + 1}: {similarity}%")

                if similarity >= 69:
                    time.sleep(1.5)  # Add a delay of 1 second before scrolling
                    current_line_index += 1  # Move to the next line
                    script_display.yview_scroll(1, "units")  # Scroll by one line
                    print(f"Scrolling to line {current_line_index}")  # Log scrolling event

                # Store the transcription for output
                transcription_log.append(current_transcription)

                # Check if all lines have been matched
                if current_line_index >= len(script_lines):
                    with open("transcription_output.txt", "w") as f:
                        f.write("\n".join(transcription_log))
                    print("Transcription complete. Output saved to transcription_output.txt.")
                    root.quit()  # Close the GUI on completion
                    break

                # Update last transcribed text
                last_transcribed_text = current_transcription

            previous_overlap = current_chunk[-overlap_size:]
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def start_gui():
    root = tk.Tk()
    root.title("Script Viewer with Scrolling")

    # Script entry widget
    script_label = tk.Label(root, text="Enter your script:")
    script_label.pack(pady=5)
    script_entry = tk.Text(root, wrap=tk.WORD, width=65, height=5, font=("Arial", 12))
    script_entry.pack(padx=10, pady=10)

    # Script display widget
    script_display_label = tk.Label(root, text="Script Viewer:")
    script_display_label.pack(pady=5)
    script_display = tk.Text(root, wrap=tk.WORD, width=80, height=4, font=("Arial", 8), state=tk.DISABLED)
    script_display.pack(padx=10, pady=10)

    def start_transcription():
        entered_script = script_entry.get("1.0", tk.END).strip()
        script_lines = split_script_into_lines(entered_script)

        # Display the script lines in the viewer
        script_display.config(state=tk.NORMAL)
        script_display.delete("1.0", tk.END)
        for line in script_lines:
            script_display.insert(tk.END, line + "\n")
        script_display.config(state=tk.DISABLED)

        # Start transcription thread with scrolling logic
        transcription_thread = threading.Thread(target=transcribe_stream, args=(script_display, entered_script, root))
        transcription_thread.daemon = True
        transcription_thread.start()

    # Start button
    start_button = tk.Button(root, text="Start Transcription", command=start_transcription)
    start_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
