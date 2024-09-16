from flask import Flask, request, jsonify
import whisper
import io
import librosa
import numpy as np
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
