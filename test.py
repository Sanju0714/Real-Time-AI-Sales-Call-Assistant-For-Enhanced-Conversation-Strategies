import whisper

#model = whisper.load_model("tiny")
model = whisper.load_model("small")
result = model.transcribe("test.wav")
print("Transcription:", result["text"])
