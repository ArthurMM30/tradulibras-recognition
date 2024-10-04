from playsound import playsound

class Talks:
    def __init__(self):
        pass

    def play(self, word):
        if(word == "eu"):
            playsound("utils/audios/eu.mp3")
        elif(word == "você"):
            playsound("utils/audios/voce.mp3")
        elif(word == "amor"):
            playsound("utils/audios/amor.mp3")

        