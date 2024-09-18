import assemblyai as aai

aai.settings.api_key = "40cf949337fe4561a9fa11f4bb2bc3b3"
FILE_URL = './speech_test.mp3'

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    print(transcript.text)
