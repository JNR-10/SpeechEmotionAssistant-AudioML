"""Gemini LLM Service for emotion-aware responses."""

import os
import google.generativeai as genai
from typing import Optional


class GeminiService:
    """Service for generating emotion-aware responses using Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_configured = False
        
        if self.api_key:
            self.configure(self.api_key)
    
    def configure(self, api_key: str):
        """Configure Gemini with API key."""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_configured = True
            print("✓ Gemini LLM configured successfully")
        except Exception as e:
            print(f"⚠ Failed to configure Gemini: {e}")
            self.is_configured = False
    
    # Language names for prompts
    LANGUAGE_NAMES = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese', 'ja': 'Japanese',
        'ko': 'Korean', 'hi': 'Hindi', 'ar': 'Arabic', 'ru': 'Russian'
    }
    
    def generate_response(self, user_text: str, emotion: str, confidence: float = 0.5, 
                          language: str = 'en') -> str:
        """Generate an emotion-aware response using Gemini.
        
        Args:
            user_text: What the user said
            emotion: Detected emotion (happy, sad, angry, etc.)
            confidence: Confidence score of emotion detection (0-1)
            language: Language code for response (en, es, fr, etc.)
        
        Returns:
            Generated response string
        """
        if not self.is_configured or self.model is None:
            return self._fallback_response(emotion, language)
        
        # Build emotion-aware prompt
        prompt = self._build_prompt(user_text, emotion, confidence, language)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠ Gemini API error: {e}")
            return self._fallback_response(emotion, language)
    
    def _build_prompt(self, user_text: str, emotion: str, confidence: float, 
                      language: str = 'en') -> str:
        """Build the prompt for Gemini."""
        emotion_context = {
            'happy': "The user sounds happy and cheerful. Match their positive energy.",
            'sad': "The user sounds sad. Be gentle, empathetic, and supportive.",
            'angry': "The user sounds frustrated or angry. Stay calm, acknowledge their feelings, and be helpful.",
            'fearful': "The user sounds anxious or fearful. Be reassuring and calming.",
            'neutral': "The user sounds neutral. Be friendly and helpful.",
            'calm': "The user sounds calm and relaxed. Maintain a pleasant conversation.",
            'surprised': "The user sounds surprised. Engage with their curiosity.",
            'disgust': "The user sounds displeased. Be understanding and try to help."
        }
        
        context = emotion_context.get(emotion, emotion_context['neutral'])
        confidence_note = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        lang_name = self.LANGUAGE_NAMES.get(language, 'English')
        
        # Add language instruction
        lang_instruction = ""
        if language != 'en':
            lang_instruction = f"\n6. IMPORTANT: Respond in {lang_name} language only"
        
        prompt = f"""You are an empathetic voice assistant that responds based on the user's emotional state.

DETECTED EMOTION: {emotion} (confidence: {confidence_note})
EMOTIONAL CONTEXT: {context}
RESPONSE LANGUAGE: {lang_name}

USER SAID: "{user_text}"

Instructions:
1. Respond naturally and conversationally (2-3 sentences max)
2. Acknowledge their emotional state subtly without being patronizing
3. Be helpful and address what they said
4. Keep the response concise - it will be spoken aloud
5. Don't mention that you're an AI or that you detected their emotion explicitly{lang_instruction}

Response:"""
        
        return prompt
    
    def _fallback_response(self, emotion: str, language: str = 'en') -> str:
        """Fallback responses when Gemini is unavailable."""
        # English fallbacks
        fallbacks_en = {
            'happy': "That's wonderful! I'm glad you're in good spirits. How can I help you today?",
            'sad': "I'm here for you. Is there anything I can help with or would you like to talk?",
            'angry': "I understand. Let's work through this together. What can I do to help?",
            'fearful': "It's okay, I'm here to help. Take your time and let me know what you need.",
            'neutral': "I'm here to help. What would you like to know?",
            'calm': "It's nice to chat with you. How can I assist you today?",
            'surprised': "Oh! What's got your attention? I'd love to hear about it.",
            'disgust': "I understand something's bothering you. How can I help make things better?"
        }
        
        # Spanish fallbacks
        fallbacks_es = {
            'happy': "¡Qué maravilloso! Me alegra que estés de buen humor. ¿En qué puedo ayudarte?",
            'sad': "Estoy aquí para ti. ¿Hay algo en lo que pueda ayudarte?",
            'angry': "Entiendo. Trabajemos juntos en esto. ¿Qué puedo hacer para ayudar?",
            'fearful': "Está bien, estoy aquí para ayudar. Tómate tu tiempo.",
            'neutral': "Estoy aquí para ayudar. ¿Qué te gustaría saber?",
            'calm': "Es agradable charlar contigo. ¿Cómo puedo ayudarte hoy?",
            'surprised': "¡Oh! ¿Qué ha llamado tu atención?",
            'disgust': "Entiendo que algo te molesta. ¿Cómo puedo ayudar?"
        }
        
        # French fallbacks
        fallbacks_fr = {
            'happy': "C'est merveilleux! Je suis content que vous soyez de bonne humeur. Comment puis-je vous aider?",
            'sad': "Je suis là pour vous. Y a-t-il quelque chose que je puisse faire?",
            'angry': "Je comprends. Travaillons ensemble. Comment puis-je aider?",
            'fearful': "Tout va bien, je suis là pour aider. Prenez votre temps.",
            'neutral': "Je suis là pour aider. Que souhaitez-vous savoir?",
            'calm': "C'est agréable de discuter avec vous. Comment puis-je vous aider?",
            'surprised': "Oh! Qu'est-ce qui a attiré votre attention?",
            'disgust': "Je comprends que quelque chose vous dérange. Comment puis-je aider?"
        }
        
        # Select language fallbacks
        fallbacks_map = {
            'en': fallbacks_en,
            'es': fallbacks_es,
            'fr': fallbacks_fr
        }
        
        fallbacks = fallbacks_map.get(language, fallbacks_en)
        return fallbacks.get(emotion, fallbacks['neutral'])
    
    def chat(self, user_text: str, emotion: str, confidence: float = 0.5, 
             history: list = None) -> str:
        """Chat with context history (for multi-turn conversations).
        
        Args:
            user_text: Current user message
            emotion: Detected emotion
            confidence: Emotion confidence
            history: List of previous messages [{"role": "user/assistant", "text": "..."}]
        
        Returns:
            Generated response
        """
        if not self.is_configured or self.model is None:
            return self._fallback_response(emotion)
        
        # Build conversation history
        history_text = ""
        if history:
            for msg in history[-5:]:  # Last 5 messages for context
                role = "User" if msg.get('role') == 'user' else "Assistant"
                history_text += f"{role}: {msg.get('text', '')}\n"
        
        prompt = f"""You are an empathetic voice assistant.

CURRENT EMOTION: {emotion} (confidence: {confidence:.0%})

CONVERSATION HISTORY:
{history_text}
User: {user_text}

Respond naturally in 2-3 sentences, acknowledging their emotional state subtly:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠ Gemini API error: {e}")
            return self._fallback_response(emotion)


# Singleton instance
_gemini_service = None

def get_gemini_service(api_key: Optional[str] = None) -> GeminiService:
    """Get or create the Gemini service singleton."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService(api_key)
    elif api_key and not _gemini_service.is_configured:
        _gemini_service.configure(api_key)
    return _gemini_service


if __name__ == "__main__":
    # Test the service
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_service.py <your_gemini_api_key>")
        print("Or set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    api_key = sys.argv[1]
    service = GeminiService(api_key)
    
    # Test responses
    test_cases = [
        ("I'm so excited about my new job!", "happy", 0.85),
        ("I've been feeling really down lately", "sad", 0.72),
        ("This is so frustrating, nothing works!", "angry", 0.68),
        ("What's the weather like today?", "neutral", 0.55),
    ]
    
    print("\n" + "=" * 60)
    print("Testing Gemini Emotion-Aware Responses")
    print("=" * 60 + "\n")
    
    for text, emotion, conf in test_cases:
        print(f"User ({emotion}, {conf:.0%}): {text}")
        response = service.generate_response(text, emotion, conf)
        print(f"Assistant: {response}\n")
