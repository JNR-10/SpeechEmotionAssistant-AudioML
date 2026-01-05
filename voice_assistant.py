"""Emotion-aware Voice Assistant."""

import os
import random
import numpy as np
import speech_recognition as sr
import pyttsx3
import librosa
import tempfile
import wave
import threading
import queue
from datetime import datetime

from data_loader import extract_mel_spectrogram
from model import EmotionRecognizer
from config import (
    MODEL_PATH, ASSISTANT_RESPONSES, SAMPLE_RATE, 
    DURATION, EMOTIONS
)


class VoiceAssistant:
    """Emotion-aware voice assistant that responds based on detected emotion."""
    
    def __init__(self, model_name='emotion_model_cnn'):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Load emotion recognition model
        self.emotion_model = EmotionRecognizer(model_type='cnn')
        try:
            self.emotion_model.load(name=model_name)
            print("✓ Emotion recognition model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load emotion model: {e}")
            print("  Please train the model first using: python train.py")
            self.emotion_model = None
        
        # State
        self.is_listening = False
        self.current_emotion = 'neutral'
        self.emotion_history = []
        
        # Adjust for ambient noise
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✓ Microphone calibrated")
    
    def speak(self, text):
        """Convert text to speech."""
        print(f"🤖 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """Listen for audio input and return the audio data."""
        print("🎤 Listening...")
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            return audio
        except sr.WaitTimeoutError:
            print("⏱ Listening timed out")
            return None
        except Exception as e:
            print(f"❌ Error listening: {e}")
            return None
    
    def transcribe(self, audio):
        """Transcribe audio to text using Google Speech Recognition."""
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def detect_emotion(self, audio):
        """Detect emotion from audio."""
        if self.emotion_model is None or self.emotion_model.model is None:
            return 'neutral', 0.5
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                f.write(audio.get_wav_data())
            
            # Extract features
            mel_spec = extract_mel_spectrogram(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            if mel_spec is None:
                return 'neutral', 0.5
            
            # Predict emotion
            emotions, confidences, probs = self.emotion_model.predict(mel_spec)
            
            emotion = emotions[0]
            confidence = confidences[0]
            
            # Update emotion history
            self.emotion_history.append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Keep only last 10 emotions
            self.emotion_history = self.emotion_history[-10:]
            
            return emotion, confidence
            
        except Exception as e:
            print(f"⚠ Emotion detection error: {e}")
            return 'neutral', 0.5
    
    def get_response(self, emotion, user_text=None):
        """Get an appropriate response based on detected emotion."""
        # Get emotion-specific responses
        responses = ASSISTANT_RESPONSES.get(emotion, ASSISTANT_RESPONSES['neutral'])
        base_response = random.choice(responses)
        
        # If we have user text, we could integrate with an LLM here
        # For now, just return the emotion-aware response
        return base_response
    
    def process_command(self, text, emotion):
        """Process user command and generate response."""
        text_lower = text.lower() if text else ""
        
        # Simple command handling
        if any(word in text_lower for word in ['bye', 'goodbye', 'exit', 'quit', 'stop']):
            return "Goodbye! Take care!", True
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            greetings = {
                'happy': "Hello! You sound cheerful today!",
                'sad': "Hello there. I'm here if you need to talk.",
                'angry': "Hello. Let's take a deep breath together.",
                'fearful': "Hello. Don't worry, I'm here to help.",
                'neutral': "Hello! How can I help you today?",
                'calm': "Hello! You seem relaxed today.",
                'surprised': "Hello! Something exciting happening?",
                'disgust': "Hello. What's on your mind?"
            }
            return greetings.get(emotion, "Hello!"), False
        
        if 'how are you' in text_lower:
            return "I'm doing well, thank you for asking! How can I assist you?", False
        
        if any(word in text_lower for word in ['emotion', 'feeling', 'mood']):
            return f"Based on your voice, you seem to be feeling {emotion}.", False
        
        if 'help' in text_lower:
            return ("I can help you with various tasks. I also detect your emotions "
                   "from your voice and respond accordingly. Try asking me something!"), False
        
        if any(word in text_lower for word in ['time', 'date']):
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%B %d, %Y')}.", False
        
        if any(word in text_lower for word in ['weather', 'temperature']):
            return ("I don't have access to weather data right now, but I hope "
                   "you're having a pleasant day!"), False
        
        if any(word in text_lower for word in ['joke', 'funny']):
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why did the AI go to therapy? It had too many deep issues!",
                "What's a computer's favorite snack? Microchips!",
                "Why was the JavaScript developer sad? Because he didn't Node how to Express himself!"
            ]
            return random.choice(jokes), False
        
        # Default: emotion-aware response
        return self.get_response(emotion, text), False
    
    def run(self):
        """Main loop for the voice assistant."""
        print("\n" + "=" * 60)
        print("🎙️  EMOTION-AWARE VOICE ASSISTANT")
        print("=" * 60)
        print("I can detect your emotions from your voice and respond accordingly.")
        print("Say 'goodbye' to exit.\n")
        
        self.speak("Hello! I'm your emotion-aware assistant. How can I help you today?")
        
        self.is_listening = True
        
        while self.is_listening:
            # Listen for audio
            audio = self.listen()
            
            if audio is None:
                continue
            
            # Detect emotion from audio
            emotion, confidence = self.detect_emotion(audio)
            self.current_emotion = emotion
            print(f"😊 Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            # Transcribe audio to text
            text = self.transcribe(audio)
            
            if text is None:
                self.speak("I didn't catch that. Could you please repeat?")
                continue
            
            # Process command and get response
            response, should_exit = self.process_command(text, emotion)
            
            # Speak response
            self.speak(response)
            
            if should_exit:
                self.is_listening = False
        
        print("\n👋 Assistant stopped.")
    
    def get_emotion_summary(self):
        """Get a summary of detected emotions."""
        if not self.emotion_history:
            return "No emotions detected yet."
        
        emotion_counts = {}
        for entry in self.emotion_history:
            emotion = entry['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        summary = "Emotion history:\n"
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            summary += f"  - {emotion}: {count} times\n"
        
        return summary


class VoiceAssistantAPI:
    """API wrapper for the voice assistant (for web interface)."""
    
    AVAILABLE_MODELS = {
        'cnn': 'emotion_model_cnn',
        'cnn_lstm': 'emotion_model_cnn_lstm',
        'lstm': 'emotion_model_lstm'
    }
    
    def __init__(self, model_name='emotion_model_cnn'):
        self.current_model_type = 'cnn'
        self.emotion_model = None
        self.model_loaded = False
        self.emotion_history = []
        
        # Load default model
        self.load_model('cnn')
    
    def load_model(self, model_type):
        """Load a specific model type (cnn, cnn_lstm, or lstm)."""
        if model_type not in self.AVAILABLE_MODELS:
            return False, f"Unknown model type: {model_type}"
        
        model_name = self.AVAILABLE_MODELS[model_type]
        
        try:
            self.emotion_model = EmotionRecognizer(model_type=model_type)
            self.emotion_model.load(name=model_name)
            self.current_model_type = model_type
            self.model_loaded = True
            print(f"✓ Loaded model: {model_type.upper()}")
            return True, f"Model {model_type.upper()} loaded successfully"
        except Exception as e:
            print(f"Could not load {model_type} model: {e}")
            self.model_loaded = False
            return False, str(e)
    
    def get_available_models(self):
        """Get list of available models."""
        return list(self.AVAILABLE_MODELS.keys())
    
    def analyze_audio(self, audio_path, language='en-US'):
        """Analyze audio file and return emotion + transcription."""
        result = {
            'emotion': 'neutral',
            'confidence': 0.5,
            'transcription': None,
            'response': None
        }
        
        # Detect emotion
        if self.model_loaded:
            try:
                mel_spec = extract_mel_spectrogram(audio_path)
                if mel_spec is not None:
                    emotions, confidences, _ = self.emotion_model.predict(mel_spec)
                    result['emotion'] = emotions[0]
                    result['confidence'] = float(confidences[0])
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        # Transcribe with language support
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                result['transcription'] = recognizer.recognize_google(audio, language=language)
        except Exception as e:
            print(f"Transcription error: {e}")
        
        # Generate response
        responses = ASSISTANT_RESPONSES.get(result['emotion'], ASSISTANT_RESPONSES['neutral'])
        result['response'] = random.choice(responses)
        
        # Update history
        self.emotion_history.append({
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def analyze_audio_bytes(self, audio_bytes, sample_rate=16000):
        """Analyze audio from bytes."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            # Write WAV header and data
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_bytes)
        
        try:
            result = self.analyze_audio(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
