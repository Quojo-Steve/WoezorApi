from flask import Flask, request, jsonify
import whisper
import io
import librosa
import numpy as np
from deep_translator import GoogleTranslator
from flasgger import Swagger

app = Flask(__name__)
model = whisper.load_model("base")  # Choose a suitable model size
swagger = Swagger(app)  # Add Swagger UI to your app

@app.route('/', methods=['GET'])
def getsomething():
    """
    Health check route
    ---
    responses:
      200:
        description: Returns working status
    """
    response = "working......"
    return jsonify(response), 200 

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate text from one language to another
    ---
    parameters:
      - name: body
        in: body
        schema:
          type: object
          required:
            - text
            - language_from
            - language_to
          properties:
            text:
              type: string
              description: Text to be translated
            language_from:
              type: string
              description: Language code of the input text (e.g., 'en')
            language_to:
              type: string
              description: Language code for the translation (e.g., 'es')
    responses:
      200:
        description: Translation successful
        schema:
          type: object
          properties:
            translated_text:
              type: string
              description: Translated text
      400:
        description: Missing text or language code
      500:
        description: Internal server error
    """
    data = request.json
    if 'text' not in data or 'language_to' not in data or 'language_from' not in data:
        return jsonify({'error': 'Missing text or language code'}), 400
    text_to_translate = data['text']
    current_language = data['language_from']
    target_language = data['language_to']
    try:
        translator = GoogleTranslator(source=current_language, target=target_language)
        translated_text = translator.translate(text_to_translate)
        return jsonify({'translated_text': translated_text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file to text
    ---
    parameters:
      - name: file
        in: formData
        type: file
        description: Audio file to be transcribed (wav, mp3, etc.)
    responses:
      200:
        description: Transcription successful
        schema:
          type: object
          properties:
            text:
              type: string
              description: Transcribed text
      400:
        description: No file part or no selected file
      500:
        description: Internal server error
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Convert FileStorage to bytes
        audio_bytes = file.read()
        
        # Load audio file into a NumPy array
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # Use Whisper's transcribe method directly
        result = model.transcribe(audio)
        
        return jsonify({'text': result['text']})
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
