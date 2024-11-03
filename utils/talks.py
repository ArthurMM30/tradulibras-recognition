import time
import pygame
from unidecode import unidecode

from playsound import playsound
# from unidecode import unidecode 
from gtts import gTTS
class Talks:
    def __init__(self):
        pass
    
    @staticmethod
    def play(word, isSpelling):
        pygame.mixer.init()

        
        if(not isSpelling):
            word_path = f"utils/audios/{unidecode(word)}.mp3"

            pygame.mixer.music.load(word_path)
            pygame.mixer.music.play()
        else:
            word_path = f"utils/audios/spelling.mp3"
            criarAudio = gTTS(text=word, lang='pt-br', slow=False)
            criarAudio.save(word_path)
            time.sleep(0.1) 
            pygame.mixer.music.load(word_path)
            pygame.mixer.music.play()
            # playsound(word_path)
            # subprocess.run(['start', '/wait', word_path], shell=True)

        