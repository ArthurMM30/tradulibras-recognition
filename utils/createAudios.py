from gtts import gTTS
import sys
import os
from unidecode import unidecode
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repository.signsDescription import SignsDescriptionClient


# db = SignsDescriptionClient()
# words = db.getAllWords()

# for word in words:
myobj = gTTS(text="bananas", lang='pt-br', slow=False)
myobj.save(f"utils/audios/{unidecode("bananas")}.mp3")