"""
Voice Detection API - Flask Application (HuggingFace Spaces Version)
Accepts Base64-encoded MP3 audio and returns AI vs Human classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import os
import logging
from datetime import datetime

# Import the detector
from detector import HybridEnsembleDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load API key from environment variable (HuggingFace Secrets)
API_KEY = os.environ.get('API_KEY', 'sk_test_123456789')
logger.info(f"API initialized with key: {API_KEY[:10]}...")

# Initialize the detector globally (load models once at startup)
logger.info("Loading AI detection models...")
detector = None

def init_detector():
    """Initialize the detector with models"""
    global detector
    try:
        detector = HybridEnsembleDetector(
            deepfake_model_path="garystafford/wav2vec2-deepfake-voice-detector",
            whisper_model_path="openai/whisper-base",
            physics_weight=0.4,
            dl_weight=0.6,
            use_local_deepfake_model=False,
            use_local_whisper_model=False,
            max_audio_duration=30,
            load_whisper=False,  # API uses client-provided language; skip Whisper to save GPU memory and startup time
        )
        logger.info("✅ Detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {str(e)}")
        return False

# Initialize detector at startup
if not init_detector():
    logger.warning("⚠️ API starting without detector - models will be loaded on first request")


# ==========================================================
# AUTHENTICATION DECORATOR
# ==========================================================
def require_api_key(f):
    """Decorator to validate API key from request headers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers
        provided_key = request.headers.get('x-api-key')
        
        if not provided_key:
            logger.warning(f"Request without API key from {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Missing API key. Please provide 'x-api-key' in request headers."
            }), 401
        
        if provided_key != API_KEY:
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Invalid API key"
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


# ==========================================================
# ROOT ENDPOINT (HuggingFace Spaces Homepage)
# ==========================================================
@app.route('/', methods=['GET'])
def home():
    """Root endpoint - API information"""
    return jsonify({
        "name": "Voice Detection API",
        "version": "1.0.0",
        "description": "AI-powered voice detection system for identifying AI-generated vs human voices",
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "detection": "/api/voice-detection"
        },
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "authentication": "Required - use 'x-api-key' header",
        "documentation": "See README for full API documentation"
    }), 200


# ==========================================================
# HEALTH CHECK ENDPOINT
# ==========================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (no authentication required)"""
    return jsonify({
        "status": "healthy",
        "service": "Voice Detection API",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": detector is not None,
        "platform": "HuggingFace Spaces"
    }), 200


# ==========================================================
# EVALUATION ENDPOINT: /detect (hackathon evaluator format)
# Returns exactly: status, classification, confidenceScore
# ==========================================================
@app.route('/detect', methods=['POST'])
@require_api_key
def detect():
    """
    Hackathon evaluation endpoint. Request/response format per evaluation guide.
    Request: { "language": "English", "audioFormat": "mp3", "audioBase64": "..." }
    Response: { "status": "success", "classification": "HUMAN"|"AI_GENERATED", "confidenceScore": 0.0-1.0 }
    """
    global detector
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400

        data = request.get_json()
        required_fields = ['language', 'audioFormat', 'audioBase64']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing)}"}), 400

        if not data.get('audioBase64') or len(data['audioBase64']) < 100:
            return jsonify({"status": "error", "message": "Invalid or empty audio data"}), 400

        if str(data.get('audioFormat', '')).lower() != 'mp3':
            return jsonify({"status": "error", "message": "Only MP3 audio format is supported"}), 400

        if detector is None:
            if not init_detector():
                return jsonify({"status": "error", "message": "Failed to load AI detection models. Please try again later."}), 503

        result = detector.analyze(data['audioBase64'], input_type="base64")

        if result['status'] != 'success':
            return jsonify({"status": "error", "message": result.get('error', 'Analysis failed')}), 500

        raw_score = float(result['confidenceScore'])
        raw_score = max(0.0, min(1.0, raw_score))
        classification = result['classification']
        # confidenceScore = confidence in the prediction (0-1)
        if classification == 'AI_GENERATED':
            confidence_score = raw_score
        else:
            confidence_score = 1.0 - raw_score
        confidence_score = max(0.0, min(1.0, round(confidence_score, 2)))

        response = {
            "status": "success",
            "classification": classification,
            "confidenceScore": confidence_score
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in /detect: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error"}), 500


# ==========================================================
# MAIN VOICE DETECTION ENDPOINT (extended response)
# ==========================================================
@app.route('/api/voice-detection', methods=['POST'])
@require_api_key
def voice_detection():
    """
    Main voice detection endpoint
    
    Expected JSON Body:
    {
        "language": "Tamil" | "English" | "Hindi" | "Malayalam" | "Telugu",
        "audioFormat": "mp3",
        "audioBase64": "base64_encoded_audio_string"
    }
    
    Returns:
    {
        "status": "success",
        "language": "Tamil",
        "classification": "AI_GENERATED" | "HUMAN",
        "confidenceScore": 0.0-1.0,
        "explanation": "..."
    }
    """
    global detector
    
    try:
        # Validate Content-Type
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['language', 'audioFormat', 'audioBase64']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Validate language
        supported_languages = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
        if data['language'] not in supported_languages:
            return jsonify({
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(supported_languages)}"
            }), 400
        
        # Validate audio format
        if data['audioFormat'].lower() != 'mp3':
            return jsonify({
                "status": "error",
                "message": "Only MP3 audio format is supported"
            }), 400
        
        # Validate base64 string
        audio_base64 = data['audioBase64']
        if not audio_base64 or len(audio_base64) < 100:
            return jsonify({
                "status": "error",
                "message": "Invalid or empty audio data"
            }), 400
        
        # Initialize detector if not already loaded
        if detector is None:
            logger.info("Lazy loading detector on first request...")
            if not init_detector():
                return jsonify({
                    "status": "error",
                    "message": "Failed to load AI detection models. Please try again later."
                }), 503
        
        # Log request
        logger.info(f"Processing voice detection request for language: {data['language']}")
        
        # Analyze audio
        result = detector.analyze(audio_base64, input_type="base64")
        
        # Check if analysis was successful
        if result['status'] != 'success':
            error_msg = result.get('error', 'Unknown error during analysis')
            logger.error(f"Analysis failed: {error_msg}")
            return jsonify({
                "status": "error",
                "message": f"Audio analysis failed: {error_msg}"
            }), 500
        
        # Prepare response (API compliant format - NO DEBUG INFO in production)
        response = {
            "status": "success",
            "language": data['language'],  # Use requested language from input
            "classification": result['classification'],
            "confidenceScore": result['confidenceScore'],
            "explanation": result['explanation']
        }
        
        logger.info(f"✅ Analysis complete: {result['classification']} (confidence: {result['confidenceScore']})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in voice_detection: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error occurred during processing"
        }), 500


# ==========================================================
# ERROR HANDLERS
# ==========================================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        "status": "error",
        "message": "Method not allowed for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


# ==========================================================
# RUN APPLICATION
# ==========================================================
if __name__ == '__main__':
    # HuggingFace Spaces uses port 7860
    port = int(os.environ.get('PORT', 7860))
    
    # Run the app
    logger.info(f"🚀 Starting Voice Detection API on port {port}")
    logger.info(f"📍 Evaluation endpoint: http://0.0.0.0:{port}/detect")
    logger.info(f"📍 Extended endpoint: http://0.0.0.0:{port}/api/voice-detection")
    logger.info(f"🔑 API Key: {API_KEY}")
    logger.info(f"🌐 Platform: HuggingFace Spaces")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Always False in production
    )