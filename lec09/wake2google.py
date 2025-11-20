import pyaudio
import json
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import subprocess  # Import subprocess to call aplay

# Initialize the Vosk model for speech recognition
model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Set up the microphone stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
stream.start_stream()

# Initialize SpeechRecognition recognizer
sr_recognizer = sr.Recognizer()

def listen_for_hotword():
    print("Listening for hotword...")
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            # print("Detected Speech: ", result['text'])

            # Trigger speech recognition when hotword is detected (e.g., "hello")
            if 'hello' in result['text'].lower():
                print("Hotword 'hello' detected. Triggering Speech Recognition...")
                
                # Play the 'ding' sound using aplay
                subprocess.run(['aplay', "ding.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Activate SpeechRecognition for further recognition after detecting hotword
                with sr.Microphone(sample_rate = 16000) as source:
                    # sr_recognizer.adjust_for_ambient_noise(source, duration=1)
                    print("Say something...")
                    audio = sr_recognizer.listen(source, phrase_time_limit=5)
                    try:
                        # Recognize speech using Google's API or offline recognizers
                        print("You said: " + sr_recognizer.recognize_google(audio, language='zh-TW'))
                        print("Listening for hotword...")
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                        print("Listening for hotword...")
                    except sr.RequestError as e:
                        print("Error with the speech service; {0}".format(e))

# Start listening for the hotword
try:
    listen_for_hotword()
except KeyboardInterrupt:
    print("Exit")
