from flask import Flask, request
from photomath import PhotoMath
from google.cloud import speech
from google.oauth2.service_account import Credentials
import wave

app = Flask(__name__)
creds = Credentials.from_service_account_file("creds.json")
client = speech.SpeechClient(credentials=creds)


@app.route("/", methods=['POST'])
def index():
    image = request.files['file']
    data = image.stream.read()
    photo = PhotoMath()
    j = photo.request(data)
    if not j:
        return {"success": False}

    return j

@app.route("/audio", methods=['POST'])
def audio():
    pcm = request.data
    with wave.open("temp.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)

    with open("temp.wav", "rb") as f:
        audio_content = f.read()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        sample_rate_hertz=16000
    )

    # Transcribes the audio into text
    audio = speech.RecognitionAudio(content=audio_content)
    response = client.recognize(config=config, audio=audio)
    if len(response.results) == 0 or len(response.results[0].alternatives) == 0:
        return "None"

    return response.results[0].alternatives[0].transcript


if __name__ == "__main__":
    app.run(host="::", port=80, debug=True)