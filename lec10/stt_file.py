import speech_recognition as sr

#obtain audio from the microphone
r=sr.Recognizer() 

myvoice = sr.AudioFile('google.wav')
with myvoice as source:
    print("Use audio file as input!")
    audio = r.record(source)

# recognize speech using Google Speech Recognition 
try:
    print("Google Speech Recognition thinks you said:")
    print(r.recognize_google(audio, language='zh-TW'))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))