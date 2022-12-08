from gtts import gTTS
import os

tts = gTTS(text='你好我是谷歌小姐', lang='zh-TW')
tts.save('hello_tw.mp3')
os.system('omxplayer -o local -p hello_tw.mp3 > /dev/null 2>&1')
