from gtts import gTTS

words = ["eu", "amor", "voce"]

for word in words:
    myobj = gTTS(text=word, lang='pt-br', slow=False)
    myobj.save("utils/audios/"+word+".mp3")