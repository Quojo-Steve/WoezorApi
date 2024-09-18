from flask import Flask, request, jsonify
from flasgger import Swagger
import assemblyai as aai
import os

app = Flask(__name__)
swagger = Swagger(app)  # Add Swagger UI to your app

# Set AssemblyAI API key
aai.settings.api_key = "40cf949337fe4561a9fa11f4bb2bc3b3"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Health check route
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

# Transcription route
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file to text
    ---
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
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
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    try:
        # Use AssemblyAI to transcribe the file
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)

        # Check for errors in the transcription process
        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({'error': transcript.error}), 500

        # Return the transcribed text
        return jsonify({'text': transcript.text}), 200

    except Exception as e:
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
