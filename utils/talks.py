from playsound import playsound
from unidecode import unidecode 
from gtts import gTTS
class Talks:
    def __init__(self):
        pass
    
    @staticmethod
    def play(word, isSpelling = False):
        if(isSpelling == True):
            obj = gTTS(text=word, lang='pt-br', slow=False)
            obj.save(f"utils/audios/{unidecode(word)}.mp3")
            
        playsound("utils/audios/"+unidecode(word)+".mp3")
        

        