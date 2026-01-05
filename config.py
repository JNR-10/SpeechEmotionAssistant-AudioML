"""Configuration settings for Speech Emotion Recognition project."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data"
MODEL_PATH = os.path.join(BASE_DIR, "models")
FEATURES_PATH = os.path.join(BASE_DIR, "features")

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)

# Audio settings
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# RAVDESS emotion labels
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Simplified emotions (combining similar ones)
EMOTION_MAP = {
    'neutral': 'neutral',
    'calm': 'calm',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fearful': 'fearful',
    'disgust': 'disgust',
    'surprised': 'surprised'
}

# Model settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Voice Assistant responses based on emotion
ASSISTANT_RESPONSES = {
    'neutral': [
        "I'm here to help. What can I do for you?",
        "How can I assist you today?",
        "What would you like to know?"
    ],
    'calm': [
        "You seem relaxed. How can I help you today?",
        "It's nice to hear you so calm. What do you need?",
        "I'm ready to assist. What's on your mind?"
    ],
    'happy': [
        "I can hear you're in a great mood! What can I do for you?",
        "Your happiness is contagious! How can I help?",
        "Wonderful to hear you so cheerful! What do you need?"
    ],
    'sad': [
        "I sense you might be feeling down. I'm here for you. How can I help?",
        "It sounds like you're having a tough time. What can I do to help?",
        "I'm sorry you're feeling this way. Let me know how I can assist."
    ],
    'angry': [
        "I understand you might be frustrated. Let me help resolve this.",
        "I hear your frustration. How can I make things better?",
        "Let's work through this together. What do you need?"
    ],
    'fearful': [
        "It's okay, I'm here to help. What's worrying you?",
        "Don't worry, we'll figure this out together. How can I assist?",
        "I understand your concern. Let me help you with that."
    ],
    'disgust': [
        "I understand something's bothering you. How can I help?",
        "Let me know what's wrong and I'll try to help.",
        "I'm here to assist. What can I do for you?"
    ],
    'surprised': [
        "Oh! Something unexpected? Tell me more!",
        "Sounds like something caught you off guard! How can I help?",
        "I'm curious to hear what surprised you! What do you need?"
    ]
}
