from gtts import gTTS

def transcribe(input, filename, language='en'):
    audio = gTTS(text=input, lang=language, slow=False)
    audio.save(f"{filename}.mp3")