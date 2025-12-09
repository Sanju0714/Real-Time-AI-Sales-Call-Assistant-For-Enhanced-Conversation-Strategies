
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
from textblob import TextBlob
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY", "Key"))

# Google Sheets Setup
def connect_google_sheet(sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).sheet1

sheet = connect_google_sheet("Sales_Call_Analysis")

# Record audio until silence
def record_until_silence(threshold=0.03, silence_duration=3, fs=16000):
    """
    Records audio until silence is detected.
    threshold: amplitude level considered as silence
    silence_duration: seconds of silence required to stop
    fs: sampling rate
    """
    print("Recording... Speak now! (Recording will stop after silence)")
    audio_data = []
    silence_time = 0

    while True:
        # Record in chunks of 0.5s
        chunk = sd.rec(int(3 * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        audio_data.append(chunk)

        # Check if this chunk is silent
        if np.max(np.abs(chunk)) < threshold:
            silence_time += 0.5
            if silence_time >= silence_duration:
                print("Silence detected, stopping recording...")
                break
        else:
            silence_time = 0  # reset silence timer if speech detected

    return np.squeeze(np.concatenate(audio_data))

# Transcribe using Groq
def transcribe_with_groq(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    return transcription

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    if polarity > 0.2:
        sentiment = "Positive"
    elif polarity < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return polarity, subjectivity, sentiment

# Process call and save to Google Sheets
def process_call():
    # Record until silence
    audio = record_until_silence(threshold=0.03, silence_duration=3, fs=16000)

    # Save audio
    sf.write("temp.wav", audio, 16000)

    # Transcribe
    transcript = transcribe_with_groq("temp.wav")

    # Sentiment analysis
    polarity, subjectivity, sentiment = analyze_sentiment(transcript)

    print("\n--- Call Analysis ---")
    print("Transcript:", transcript)
    print(f"Sentiment: {sentiment} | Polarity: {polarity} | Subjectivity: {subjectivity}")

    #Log to Google Sheets
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        transcript,
        sentiment,
        polarity,
        subjectivity
    ])
    print("Data logged to Google Sheet successfully!")

if __name__ == "__main__":
    process_call()
