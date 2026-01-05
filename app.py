"""Flask web application for Emotion-Aware Voice Assistant."""

import os
import tempfile
import json
import base64
import wave
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from voice_assistant import VoiceAssistantAPI
from config import EMOTIONS, ASSISTANT_RESPONSES
from llm_service import get_gemini_service
from data_loader import extract_mel_spectrogram
from report_generator import get_report_generator

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize the voice assistant API
assistant = VoiceAssistantAPI()

# Initialize Gemini LLM service
gemini = get_gemini_service(os.getenv('GEMINI_API_KEY'))

# Store for streaming audio chunks
streaming_sessions = {}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


# Supported languages for speech recognition
SUPPORTED_LANGUAGES = {
    'en': {'code': 'en-US', 'name': 'English', 'flag': '🇺🇸'},
    'es': {'code': 'es-ES', 'name': 'Español', 'flag': '🇪🇸'},
    'fr': {'code': 'fr-FR', 'name': 'Français', 'flag': '🇫🇷'},
    'de': {'code': 'de-DE', 'name': 'Deutsch', 'flag': '🇩🇪'},
    'it': {'code': 'it-IT', 'name': 'Italiano', 'flag': '🇮🇹'},
    'pt': {'code': 'pt-BR', 'name': 'Português', 'flag': '🇧🇷'},
    'zh': {'code': 'zh-CN', 'name': '中文', 'flag': '🇨🇳'},
    'ja': {'code': 'ja-JP', 'name': '日本語', 'flag': '🇯🇵'},
    'ko': {'code': 'ko-KR', 'name': '한국어', 'flag': '🇰🇷'},
    'hi': {'code': 'hi-IN', 'name': 'हिन्दी', 'flag': '🇮🇳'},
    'ar': {'code': 'ar-SA', 'name': 'العربية', 'flag': '🇸🇦'},
    'ru': {'code': 'ru-RU', 'name': 'Русский', 'flag': '🇷🇺'},
}


@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get list of supported languages."""
    return jsonify({
        'languages': SUPPORTED_LANGUAGES,
        'default': 'en'
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file for emotion and transcription."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'en')
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Get language code for speech recognition
    lang_info = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES['en'])
    lang_code = lang_info['code']
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        audio_file.save(temp_path)
    
    try:
        result = assistant.analyze_audio(temp_path, language=lang_code)
        result['language'] = language
        
        # Generate LLM response if transcription available
        if result.get('transcription') and gemini.is_configured:
            llm_response = gemini.generate_response(
                result['transcription'],
                result['emotion'],
                result['confidence'],
                language=language
            )
            result['response'] = llm_response
            result['llm_enabled'] = True
        else:
            result['llm_enabled'] = False
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions."""
    return jsonify({
        'emotions': list(EMOTIONS.values()),
        'responses': ASSISTANT_RESPONSES
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get emotion detection history."""
    return jsonify({
        'history': assistant.emotion_history[-20:]
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        'status': 'online',
        'model_loaded': assistant.model_loaded,
        'current_model': assistant.current_model_type,
        'available_models': assistant.get_available_models(),
        'llm_enabled': gemini.is_configured,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/model', methods=['POST'])
def switch_model():
    """Switch the emotion recognition model."""
    data = request.get_json()
    
    if not data or 'model_type' not in data:
        return jsonify({'error': 'No model_type provided'}), 400
    
    model_type = data['model_type']
    success, message = assistant.load_model(model_type)
    
    return jsonify({
        'success': success,
        'message': message,
        'current_model': assistant.current_model_type
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models."""
    return jsonify({
        'models': assistant.get_available_models(),
        'current': assistant.current_model_type
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for text-based conversation with emotion context."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_text = data['text']
    emotion = data.get('emotion', 'neutral')
    confidence = data.get('confidence', 0.5)
    
    if gemini.is_configured:
        response = gemini.generate_response(user_text, emotion, confidence)
    else:
        # Fallback to predefined responses
        import random
        responses = ASSISTANT_RESPONSES.get(emotion, ASSISTANT_RESPONSES['neutral'])
        response = random.choice(responses)
    
    return jsonify({
        'response': response,
        'llm_enabled': gemini.is_configured
    })


@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate PDF report from session data."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No session data provided'}), 400
    
    try:
        report_gen = get_report_generator()
        
        # Build session data
        session_data = {
            'session_id': data.get('session_id', 'session_' + datetime.now().strftime('%Y%m%d_%H%M%S')),
            'emotions': data.get('emotions', []),
            'dominant_emotion': data.get('dominant_emotion', 'neutral'),
            'average_confidence': data.get('average_confidence', 0.5),
            'emotion_breakdown': data.get('emotion_breakdown', {}),
            'model_used': data.get('model_used', assistant.current_model_type),
            'language': data.get('language', 'en'),
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
        }
        
        # Generate PDF
        pdf_bytes = report_gen.generate_report(session_data)
        
        # Return as downloadable file
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        print(f"Report generation error: {e}")
        return jsonify({'error': str(e)}), 500


# WebSocket events for real-time streaming
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    streaming_sessions[session_id] = {
        'audio_buffer': b'',
        'chunk_count': 0,
        'last_emotion': 'neutral',
        'emotions': []
    }
    emit('connected', {'session_id': session_id})
    print(f"Client connected: {session_id}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    session_id = request.sid
    if session_id in streaming_sessions:
        del streaming_sessions[session_id]
    print(f"Client disconnected: {session_id}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for real-time analysis."""
    session_id = request.sid
    
    if session_id not in streaming_sessions:
        streaming_sessions[session_id] = {
            'audio_buffer': b'',
            'chunk_count': 0,
            'last_emotion': 'neutral',
            'emotions': []
        }
    
    session = streaming_sessions[session_id]
    
    # Decode base64 audio data
    try:
        audio_bytes = base64.b64decode(data['audio'])
        session['audio_buffer'] += audio_bytes
        session['chunk_count'] += 1
        
        # Analyze every ~2 seconds of audio (assuming 16kHz, 16-bit = 32000 bytes/sec)
        # We'll analyze every 64000 bytes (~2 seconds)
        if len(session['audio_buffer']) >= 64000:
            emotion, confidence = analyze_audio_buffer(
                session['audio_buffer'],
                data.get('sample_rate', 16000)
            )
            
            session['last_emotion'] = emotion
            session['emotions'].append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 20 emotions
            session['emotions'] = session['emotions'][-20:]
            
            # Emit result back to client
            emit('emotion_update', {
                'emotion': emotion,
                'confidence': confidence,
                'chunk_count': session['chunk_count']
            })
            
            # Clear buffer but keep last 0.5 seconds for continuity
            session['audio_buffer'] = session['audio_buffer'][-16000:]
            
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        emit('error', {'message': str(e)})


@socketio.on('stop_streaming')
def handle_stop_streaming():
    """Handle end of streaming session."""
    session_id = request.sid
    
    if session_id in streaming_sessions:
        session = streaming_sessions[session_id]
        emotions = session.get('emotions', [])
        
        # Calculate emotion summary
        if emotions:
            emotion_counts = {}
            for e in emotions:
                em = e['emotion']
                emotion_counts[em] = emotion_counts.get(em, 0) + 1
            
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = sum(e['confidence'] for e in emotions) / len(emotions)
        else:
            dominant_emotion = 'neutral'
            avg_confidence = 0.5
        
        emit('streaming_complete', {
            'dominant_emotion': dominant_emotion,
            'average_confidence': avg_confidence,
            'total_detections': len(emotions),
            'emotion_breakdown': emotion_counts if emotions else {}
        })
        
        # Clear session
        streaming_sessions[session_id] = {
            'audio_buffer': b'',
            'chunk_count': 0,
            'last_emotion': 'neutral',
            'emotions': []
        }


def analyze_audio_buffer(audio_buffer, sample_rate=16000):
    """Analyze audio buffer and return emotion."""
    try:
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_buffer)
        
        # Extract features and predict
        if assistant.model_loaded and assistant.emotion_model:
            mel_spec = extract_mel_spectrogram(temp_path)
            if mel_spec is not None:
                emotions, confidences, _ = assistant.emotion_model.predict(mel_spec)
                os.unlink(temp_path)
                return emotions[0], float(confidences[0])
        
        os.unlink(temp_path)
        return 'neutral', 0.5
        
    except Exception as e:
        print(f"Error in analyze_audio_buffer: {e}")
        return 'neutral', 0.5


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎙️  EMOTION-AWARE VOICE ASSISTANT - WEB INTERFACE")
    print("=" * 60)
    print(f"Model loaded: {assistant.model_loaded}")
    print(f"Gemini LLM: {'✓ Enabled' if gemini.is_configured else '✗ Disabled'}")
    print("Real-time streaming: ✓ Enabled")
    print("Starting server at http://localhost:5000")
    print("=" * 60 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
