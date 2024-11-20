import speech_recognition as sr

#obtain audio from the microphone
recognizer = sr.Recognizer()

with sr.Microphone(sample_rate = 16000) as source:
    print("Please wait. Calibrating microphone...")
    #listen for 1 seconds and create the ambient noise energy level
    # recognizer.adjust_for_ambient_noise(source, duration=1)
    # recognizer.energy_threshold = 4000
    print("Say something!")
    audio = recognizer.listen(source, phrase_time_limit=5)

# recognize speech using Google Speech Recognition
try:
    print("Google Speech Recognition thinks you said:")
    results = recognizer.recognize_google(audio, language='zh-TW')
#    results = r.recognize_google(audio, language='en-US')
    print(results)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))
    
