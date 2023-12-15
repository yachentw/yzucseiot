import speech_recognition as sr
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from gtts import gTTS
import os

options = Options()
#關閉瀏覽器跳出訊息
prefs = {
    'profile.default_content_setting_values' :
        {
        'notifications' : 2
         }
}
options.add_experimental_option('prefs',prefs)
options.add_argument("--headless")            #不開啟實體瀏覽器背景執行
options.add_argument("--incognito")           #開啟無痕模式

browser_driver = Service('/usr/lib/chromium-browser/chromedriver')
driver = webdriver.Chrome(service=browser_driver, options=options)

#obtain audio from the microphone
r=sr.Recognizer() 

with sr.Microphone(sample_rate = 48000) as source:
    print("Please wait. Calibrating microphone...") 
    #listen for 1 seconds and create the ambient noise energy level 
    # r.energy_threshold = 4000
    r.adjust_for_ambient_noise(source, duration=1) 
    print("Say something!")
    audio=r.listen(source)

# recognize speech using Google Speech Recognition 
try:
    print("Google Speech Recognition thinks you said:")
    text = r.recognize_google(audio, language='zh-TW')
    print(text)
    if "天氣" in text:
        driver.get("https://www.google.com/search?q=中壢+天氣")
        Temp = driver.find_element("id", 'wob_tm').text
#        Temp = driver.find_element_by_id('wob_tm').text
        tts = gTTS(text='現在中壢天氣' + Temp + '度', lang='zh-TW')
        tts.save('w.mp3')
        os.system('omxplayer -o local -p w.mp3 > /dev/null 2>&1')
        os.remove('w.mp3')
        print(Temp)
        driver.quit()
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))




