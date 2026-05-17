from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import io
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audio_cleaner import AudioCleaner
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load pre-trained model and processor (outside of a function for efficiency)
processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-luganda")
model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-luganda")

# Initialize audio cleaner
audio_cleaner = AudioCleaner(sr=16000)

def audio_preprocessor(audio_bytes, clean_audio=True, reduction_strength=0.5):
    """
    Preprocesses audio data by:
        - Loading the audio data from the request.
        - Cleaning audio (optional) to remove noise and enhance voice.
        - Resampling the audio to 16 kHz.
        - Converting the audio to a NumPy array.

    Args:
        audio_bytes (bytes): The audio data received in the request.
        clean_audio (bool): Whether to apply audio cleaning.
        reduction_strength (float): Noise reduction strength (0.0-1.0).
    
    Returns:
        tuple: (np.ndarray preprocessed audio, dict audio stats)
    """
    if not audio_bytes:
        raise ValueError("No audio data provided in the request.")

    # Load audio from bytes
    audio, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))
    audio_array = audio.squeeze().numpy()

    # Get audio stats before cleaning
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio = resampler(audio)
        audio_array = audio.squeeze().numpy()
    
    stats_before = audio_cleaner.get_audio_stats(audio_array)

    # Apply cleaning if requested
    if clean_audio:
        audio_array = audio_cleaner.clean_audio(
            audio_array,
            apply_noise_reduction=True,
            apply_enhancement=True,
            apply_normalization=True,
            apply_trim_silence=True,
            reduction_strength=reduction_strength
        )

    stats_after = audio_cleaner.get_audio_stats(audio_array)

    return audio_array, {
        "before_cleaning": stats_before,
        "after_cleaning": stats_after
    }

# Serve the web UI
@app.route('/')
def index():
    """Serve the main web UI."""
    return send_from_directory('.', 'index.html')

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Luganda Speech-to-Text API with Audio Cleaning"}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribes uploaded audio data using the Wav2Vec2 model.
    Optionally cleans audio for better transcription quality.

    Query Parameters:
        clean_audio (bool): Enable audio cleaning (default: true)
        reduction_strength (float): Noise reduction strength 0.0-1.0 (default: 0.5)
    
    Returns:
        dict: JSON response with transcription and audio stats
    """
    if request.method != "POST":
        return jsonify({"error": "Method not allowed. Please use POST."}), 405

    # Read audio data from the request
    try:
        audio_bytes = request.files["audio"].read()
    except KeyError:
        return jsonify({"error": "Missing 'audio' field in request form."}), 400

    # Get parameters
    clean_audio = request.args.get("clean_audio", "true").lower() == "true"
    reduction_strength = float(request.args.get("reduction_strength", "0.5"))

    # Preprocess audio
    try:
        audio_array, stats = audio_preprocessor(
            audio_bytes,
            clean_audio=clean_audio,
            reduction_strength=reduction_strength
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Prepare model input
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

    # Disable gradient calculation
    try:
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # Find the most likely character for each position
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted sequence
        prediction = processor.batch_decode(predicted_ids)[0]

        response = {
            "transcription": prediction,
            "audio_cleaning_applied": clean_audio,
            "audio_stats": stats if clean_audio else None
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route("/audio-stats", methods=["POST"])
def get_audio_stats():
    """
    Analyzes audio quality without transcription.
    
    Query Parameters:
        clean_audio (bool): Analyze after cleaning (default: false)
    
    Returns:
        dict: Audio quality metrics
    """
    try:
        audio_bytes = request.files["audio"].read()
    except KeyError:
        return jsonify({"error": "Missing 'audio' field in request form."}), 400

    clean_audio = request.args.get("clean_audio", "false").lower() == "true"

    try:
        audio_array, stats = audio_preprocessor(audio_bytes, clean_audio=clean_audio)
        return jsonify({
            "status": "success",
            "audio_stats": stats
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe-and-translate", methods=["POST"])
def transcribe_and_translate():
    """
    Transcribes audio and prepares translation metadata.
    Future integration point for translation services.
    
    Query Parameters:
        clean_audio (bool): Enable audio cleaning (default: true)
        target_language (str): Target language code (default: en)
    
    Returns:
        dict: Transcription and translation metadata
    """
    try:
        audio_bytes = request.files["audio"].read()
    except KeyError:
        return jsonify({"error": "Missing 'audio' field in request form."}), 400

    clean_audio = request.args.get("clean_audio", "true").lower() == "true"
    target_language = request.args.get("target_language", "en")

    try:
        audio_array, stats = audio_preprocessor(audio_bytes, clean_audio=clean_audio)

        # Transcribe
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Prepare translation metadata
        response = {
            "transcription": transcription,
            "source_language": "lg",  # Luganda ISO 639-1 code
            "target_language": target_language,
            "audio_cleaning_applied": clean_audio,
            "audio_stats": stats if clean_audio else None,
            "translation_status": "ready for translation service",
            "note": "Integrate with translation service to convert transcription"
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  Luganda Speech-to-Text API with Audio Cleaning")
    print("="*60)
    print("\n✅ Opening web UI at: http://localhost:5000")
    print("\n📝 Instructions:")
    print("   1. Upload your audio file (WAV, MP3, FLAC)")
    print("   2. Adjust noise reduction strength (0.3-0.7 recommended)")
    print("   3. Click 'Transcribe Audio'")
    print("   4. View your transcription and audio metrics\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
