from playsound import playsound
from unidecode import unidecode 
from gtts import gTTS
class Talks:
    def __init__(self):
        pass
    
    @staticmethod
    def play(word):            
        playsound("utils/audios/"+unidecode(word)+".mp3")
        

        