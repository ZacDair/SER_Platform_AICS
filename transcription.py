# External imports
import speech_recognition as sr

# Project level imports
# import config


# Transcribe a WAV file to text and return the result
def transcribe(audioFilePath):
    audioFile = sr.AudioFile(audioFilePath)

    # Define the speech recognising method (Google, IBM..etc)
    r = sr.Recognizer()

    # Use the audio file as the source
    with audioFile as source:
        audio = r.record(source)  # read the entire audio file

    return r.recognize_google(audio)