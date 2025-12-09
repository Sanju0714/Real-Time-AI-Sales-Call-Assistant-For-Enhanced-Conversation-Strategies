import sounddevice as sd
import numpy as np
import wave

def record_audio(filename="test.wav", duration=5, fs=16000):
    print("Recording for 5 seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait() 
    print("Recording complete.")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)       
        wf.setsampwidth(2)       
        wf.setframerate(fs)       
        wf.writeframes(recording.tobytes())

    print(f"Audio saved as {filename}")

# Record a 5-second audio clip
record_audio()
