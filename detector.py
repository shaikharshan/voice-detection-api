import torch
import librosa
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
import tempfile
import os
import soundfile as sf
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

class HybridEnsembleDetector:
    """
    Hybrid AI Voice Detection System with Language Detection
    
    Features:
    1. Physics-based acoustic analysis
    2. Deep Learning deepfake detection
    3. Language identification using Whisper (focus on Indian languages)
    4. Auto-truncation to 30 seconds for faster processing
    """
    
    def __init__(
        self, 
        deepfake_model_path="garystafford/wav2vec2-deepfake-voice-detector",
        whisper_model_path="openai/whisper-base",
        physics_weight=0.4,
        dl_weight=0.6,
        use_local_deepfake_model=False,
        use_local_whisper_model=False,
        max_audio_duration=30,  # seconds
        load_whisper=True,
    ):
        """
        Initialize the hybrid detector
        
        Args:
            deepfake_model_path: Path to deepfake detection model
            whisper_model_path: Path to Whisper model for language detection
            physics_weight: Weight for physics score (0-1)
            dl_weight: Weight for DL score (0-1)
            use_local_deepfake_model: Whether to load deepfake model from local path
            use_local_whisper_model: Whether to load Whisper from local path
            max_audio_duration: Maximum audio duration to process (seconds)
            load_whisper: If False, skip loading Whisper (saves GPU memory when language detection is not used)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # float16 on GPU (e.g. T4) = faster inference + less memory; no benefit on CPU
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.max_duration = max_audio_duration
        
        # Normalize weights
        total_weight = physics_weight + dl_weight
        self.physics_weight = physics_weight / total_weight
        self.dl_weight = dl_weight / total_weight
        
        print(f"🔧 Initializing Hybrid Detector with Language Detection")
        print(f"   Device: {self.device} (dtype: {self.torch_dtype})")
        print(f"   Physics Weight: {self.physics_weight*100:.0f}%")
        print(f"   DL Weight: {self.dl_weight*100:.0f}%")
        print(f"   Max Audio Duration: {self.max_duration}s")
        
        # --- LOAD DEEPFAKE DETECTION MODEL ---
        try:
            print(f"📥 Loading deepfake detection model from '{deepfake_model_path}'...")
            
            if use_local_deepfake_model:
                self.dl_model = AutoModelForAudioClassification.from_pretrained(
                    deepfake_model_path,
                    local_files_only=True,
                    torch_dtype=self.torch_dtype,
                )
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    deepfake_model_path,
                    local_files_only=True,
                )
            else:
                self.dl_model = AutoModelForAudioClassification.from_pretrained(
                    deepfake_model_path,
                    torch_dtype=self.torch_dtype,
                )
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(deepfake_model_path)
            
            self.dl_model.to(self.device)
            self.dl_model.eval()
            # torch.compile() not used: inductor backend requires a C++ compiler (g++), which
            # is often missing in minimal Docker/HF Spaces images and causes InvalidCxxCompiler at first run.
            self.dl_ready = True
            print("✅ Deepfake Detection Model Loaded")
            
        except Exception as e:
            print(f"⚠️  DL Model Load Failed: {e}")
            print("   Running in Physics-Only mode")
            self.dl_ready = False
            self.dl_weight = 0
            self.physics_weight = 1.0

        # --- LOAD WHISPER FOR LANGUAGE DETECTION (optional; skip if API does not use detect_language) ---
        self.lang_ready = False
        self.whisper_model = None
        self.whisper_processor = None
        self.language_map = {}
        if load_whisper:
            try:
                print(f"📥 Loading Whisper model for language detection from '{whisper_model_path}'...")
                
                if use_local_whisper_model:
                    self.whisper_processor = WhisperProcessor.from_pretrained(
                        whisper_model_path,
                        local_files_only=True,
                    )
                    self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                        whisper_model_path,
                        local_files_only=True,
                        torch_dtype=self.torch_dtype,
                    )
                else:
                    self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)
                    self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                        whisper_model_path,
                        torch_dtype=self.torch_dtype,
                    )
                
                self.whisper_model.to(self.device)
                self.whisper_model.eval()
                self.lang_ready = True
                print("✅ Whisper Language Detection Model Loaded")
                
                # Language code mapping for Indian languages and common languages
                self.language_map = {
                    'hi': 'Hindi',
                    'bn': 'Bengali', 
                    'te': 'Telugu',
                    'mr': 'Marathi',
                    'ta': 'Tamil',
                    'gu': 'Gujarati',
                    'kn': 'Kannada',
                    'ml': 'Malayalam',
                    'or': 'Odia',
                    'pa': 'Punjabi',
                    'as': 'Assamese',
                    'ur': 'Urdu',
                    'en': 'English',
                    'ne': 'Nepali',
                    'si': 'Sinhala',
                    'sa': 'Sanskrit',
                    'sd': 'Sindhi',
                    'ks': 'Kashmiri'
                }
                
            except Exception as e:
                print(f"⚠️  Whisper Model Load Failed: {e}")
                print("   Running without language detection")
                self.lang_ready = False
        else:
            print("   Skipping Whisper (load_whisper=False)")

        # --- PHYSICS ENGINE PARAMETERS ---
        self.CV_AI_THRESHOLD = 0.20
        self.CV_HUMAN_THRESHOLD = 0.32
        self.INTENSITY_MIN_STD = 0.05
        self.INTENSITY_MAX_STD = 0.15
        # Analyze only middle N seconds for physics (reduces latency ~60% vs full 30s)
        self.physics_analysis_duration = 8  # seconds; middle segment where voice is most stable

        # --- CHUNKING (MAX-POOLING) FOR LONG AUDIO ---
        self.chunk_duration_sec = 5
        self.max_chunks = 3  # If any chunk is AI, whole file is AI

        # --- CLASSIFICATION THRESHOLD (language-specific) ---
        self.default_ai_threshold = 0.55
        self.tamil_ai_threshold = 0.45  # More aggressive for Tamil TTS

        print("✅ Hybrid Detector Ready\n")

    # ==========================================================
    # HELPER: Audio Preprocessing
    # ==========================================================
    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Load and preprocess audio:
        1. Load audio
        2. Convert to mono
        3. Truncate to max_duration if needed
        4. Resample to target_sr
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            tuple: (waveform_array, sample_rate, duration, was_truncated)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Calculate duration
            duration = len(y) / sr
            was_truncated = False
            
            # Truncate if longer than max_duration
            if duration > self.max_duration:
                print(f"   ⚠️  Audio is {duration:.1f}s, truncating to {self.max_duration}s")
                max_samples = int(self.max_duration * sr)
                y = y[:max_samples]
                duration = self.max_duration
                was_truncated = True
            
            # Resample if needed
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            return y, sr, duration, was_truncated
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess audio: {str(e)}")

    def _load_audio_once(self, audio_path):
        """
        Load audio once at native sample rate and truncate to max_duration.
        Used to avoid loading the same file twice in physics + DL branches.
        Returns:
            tuple: (y, sr, duration, was_truncated)
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr
        was_truncated = False
        if duration > self.max_duration:
            max_samples = int(self.max_duration * sr)
            y = y[:max_samples]
            duration = self.max_duration
            was_truncated = True
        return y, sr, duration, was_truncated

    def _chunk_audio(self, y, sr, duration):
        """
        Split audio into fixed-duration chunks for max-pooling.
        Returns list of (y_chunk, sr, duration_chunk, was_truncated) for preloaded.
        """
        chunks = []
        chunk_len = int(self.chunk_duration_sec * sr)
        min_chunk_samples = int(1.0 * sr)  # skip chunks shorter than 1s
        for i in range(self.max_chunks):
            start = i * chunk_len
            if start >= len(y):
                break
            end = min(start + chunk_len, len(y))
            y_chunk = y[start:end]
            if len(y_chunk) < min_chunk_samples:
                break
            dur_chunk = len(y_chunk) / sr
            chunks.append((y_chunk.copy(), sr, dur_chunk, False))
        return chunks

    # ==========================================================
    # HELPER: Base64 Decoding
    # ==========================================================
    def decode_base64_audio(self, base64_string):
        """
        Decode base64 audio and save to temporary file
        
        Args:
            base64_string: Base64 encoded audio data
            
        Returns:
            str: Path to temporary audio file
        """
        try:
            # Decode base64
            audio_data = base64.b64decode(base64_string)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(audio_data)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {str(e)}")

    # ==========================================================
    # LANGUAGE DETECTION
    # ==========================================================
    def detect_language(self, audio_path):
        """
        Detect language using Whisper model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Detected language name
        """
        if not self.lang_ready:
            return "Unknown"
        
        try:
            # Load and preprocess audio for Whisper (uses 16kHz)
            # Use first 30 seconds for language detection
            audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
            
            # Process audio with Whisper processor
            input_features = self.whisper_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            input_features = input_features.to(device=self.device, dtype=self.torch_dtype)
            
            # Whisper language detection using forced_decoder_ids
            with torch.inference_mode():
                # Generate with language detection enabled
                generated_ids = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                    return_dict_in_generate=True
                )
                
                # Decode the output
                full_output = self.whisper_processor.batch_decode(
                    generated_ids.sequences,
                    skip_special_tokens=False
                )[0]
                
                # Parse language from special tokens
                # Format: <|startoftranscript|><|en|><|transcribe|>...
                detected_lang = None
                
                # Look for language tokens in the format <|xx|>
                import re
                lang_pattern = r'<\|([a-z]{2})\|>'
                matches = re.findall(lang_pattern, full_output)
                
                if matches:
                    # First match after startoftranscript is usually the language
                    for match in matches:
                        if match in self.language_map:
                            detected_lang = match
                            break
                
                if detected_lang:
                    lang_name = self.language_map.get(detected_lang, detected_lang.upper())
                    print(f"   🌐 Detected Language: {lang_name} ({detected_lang})")
                    return lang_name
                else:
                    # Fallback: if transcription successful, assume English
                    transcription = self.whisper_processor.batch_decode(
                        generated_ids.sequences,
                        skip_special_tokens=True
                    )[0]
                    
                    if len(transcription.strip()) > 0:
                        print(f"   🌐 Detected Language: English (default)")
                        return "English"
                    else:
                        return "Unknown"
                    
        except Exception as e:
            print(f"   ⚠️  Language detection error: {str(e)}")
            return "Unknown"

    # ==========================================================
    # PART A: PHYSICS ENGINE (FIXED)
    # ==========================================================
    def get_linear_score(self, val, min_val, max_val):
        """Linear interpolation for scoring"""
        if val <= min_val:
            return 1.0
        if val >= max_val:
            return 0.0
        return 1.0 - ((val - min_val) / (max_val - min_val))

    def get_physics_score(self, audio_path, preloaded=None):
        """
        Analyze audio using physics-based acoustic features.
        If preloaded is provided, use (y, sr, duration, was_truncated) instead of loading from path.
        
        Returns:
            tuple: (ai_score, method, features_dict)
        """
        try:
            if preloaded is not None:
                y, sr, duration, was_truncated = preloaded
            else:
                y, sr, duration, was_truncated = self._load_audio_once(audio_path)

            # Use middle segment only for physics (faster; voice most stable in middle)
            seg_dur = min(self.physics_analysis_duration, duration)
            half = (duration - seg_dur) / 2.0
            start_samp = int(half * sr)
            end_samp = int((half + seg_dur) * sr)
            y_physics = y[start_samp:end_samp]

            # VAD: trim silence so physics doesn't see "perfect stability" from empty segments
            y_physics_trimmed, _ = librosa.effects.trim(y_physics, top_db=20)
            if len(y_physics_trimmed) >= int(0.5 * sr):  # Keep at least 0.5s of audio
                y_physics = y_physics_trimmed

            print(f"   🔬 Running physics analysis on middle {seg_dur:.1f}s of {duration:.1f}s audio at {sr}Hz (VAD trimmed)")
            
            # Robust pitch tracking using pYIN (filters silence/noise; yin blindly estimates pitch there)
            try:
                f0, _, _ = librosa.pyin(
                    y_physics,
                    fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                    fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                    sr=sr,
                    frame_length=2048,
                )
                valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            except Exception as pitch_error:
                print(f"   ⚠️  Pitch detection failed: {pitch_error}, using fallback method")
                valid_f0 = np.array([])
            
            if len(valid_f0) < 10:  # Need at least 10 valid pitch points
                print(f"   ⚠️  Insufficient pitch data ({len(valid_f0)} points), using alternative features")
                # Fall back to non-pitch features (on segment)
                rms = librosa.feature.rms(y=y_physics)[0]
                centroid = librosa.feature.spectral_centroid(y=y_physics, sr=sr)[0]
                zcr = librosa.feature.zero_crossing_rate(y_physics)[0]
                
                feats = {
                    'pitch_cv': 0.25,  # Neutral value
                    'intensity_std': np.std(rms),
                    'freq_skew': stats.skew(centroid),
                    'zcr_std': np.std(zcr),
                    'mean_pitch': 0,
                    'std_pitch': 0,
                    'duration': duration,
                    'was_truncated': was_truncated
                }
                
                # Score based on available features
                intensity_score = self.get_linear_score(
                    feats['intensity_std'], 
                    self.INTENSITY_MIN_STD, 
                    self.INTENSITY_MAX_STD
                )
                
                zcr_score = self.get_linear_score(
                    feats['zcr_std'],
                    0.01,
                    0.08
                )
                
                skew_score = self.get_linear_score(
                    abs(feats['freq_skew']), 
                    0.1, 
                    1.0
                )
                
                # Weighted combination (no pitch)
                final_score = (intensity_score * 0.5 + zcr_score * 0.2 + skew_score * 0.3)
                
                print(f"   🔬 Physics score (no pitch): {final_score:.3f}")
                return round(final_score, 3), "Physics Analysis (Limited)", feats

            # Full analysis with pitch (on segment)
            rms = librosa.feature.rms(y=y_physics)[0]
            centroid = librosa.feature.spectral_centroid(y=y_physics, sr=sr)[0]
            
            mean_pitch = np.mean(valid_f0)
            std_pitch = np.std(valid_f0)
            
            # Calculate feature metrics
            feats = {
                'pitch_cv': std_pitch / mean_pitch if mean_pitch > 0 else 0,
                'intensity_std': np.std(rms),
                'freq_skew': stats.skew(centroid),
                'mean_pitch': mean_pitch,
                'std_pitch': std_pitch,
                'duration': duration,
                'was_truncated': was_truncated
            }

            # Individual feature scores (higher = more AI-like)
            intensity_score = self.get_linear_score(
                feats['intensity_std'], 
                self.INTENSITY_MIN_STD, 
                self.INTENSITY_MAX_STD
            )
            
            pitch_score = self.get_linear_score(
                feats['pitch_cv'], 
                self.CV_AI_THRESHOLD, 
                self.CV_HUMAN_THRESHOLD
            )
            
            skew_score = self.get_linear_score(
                abs(feats['freq_skew']), 
                0.1, 
                1.0
            )

            # Weighted combination
            W_INTENSITY = 0.40
            W_PITCH = 0.40
            W_SKEW = 0.20
            
            base_score = (
                intensity_score * W_INTENSITY + 
                pitch_score * W_PITCH + 
                skew_score * W_SKEW
            )

            # Synergy bonus: if both intensity and pitch are suspicious
            if intensity_score > 0.4 and pitch_score > 0.4:
                final_score = min(base_score + 0.15, 1.0)
            else:
                final_score = base_score

            print(f"   🔬 Physics score: {final_score:.3f} (intensity:{intensity_score:.2f}, pitch:{pitch_score:.2f})")
            return round(final_score, 3), "Physics Analysis", feats

        except Exception as e:
            print(f"   ❌ Physics analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, f"Physics Error: {str(e)}", {'duration': 0, 'was_truncated': False}

    # ==========================================================
    # PART B: DEEP LEARNING ENGINE
    # ==========================================================
    def get_dl_score(self, audio_path, preloaded=None):
        """
        Analyze audio using deep learning model.
        If preloaded is provided, use (y, sr, duration, was_truncated) and resample to 16kHz for the model.
        
        Returns:
            tuple: (ai_score, label)
        """
        if not self.dl_ready:
            return 0.0, "Model not loaded"

        try:
            if preloaded is not None:
                y, sr, duration, was_truncated = preloaded
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                waveform_np = y
                sr = 16000
            else:
                waveform_np, sr, duration, was_truncated = self.preprocess_audio(audio_path, target_sr=16000)

            # Process with feature extractor
            inputs = self.feature_extractor(
                waveform_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device and match model dtype (important for float16 on GPU)
            inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) for k, v in inputs.items()}

            # Run inference
            with torch.inference_mode():
                outputs = self.dl_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
            # Get predictions
            # Class 0: Real, Class 1: Fake
            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()
            
            # AI score is the fake probability
            ai_score = prob_fake
            
            label = "Fake/Deepfake" if prob_fake > 0.5 else "Real/Human"
            print(f"Confidence score generated by DL model:${ai_score}, label by DL model: ${label}")

            return round(ai_score, 3), label

        except Exception as e:
            print(f"   ❌ DL analysis failed: {str(e)}")
            return 0.0, f"DL Error: {str(e)}"

    # ==========================================================
    # PART C: EXPLANATION GENERATOR
    # ==========================================================
    def generate_explanation(self, final_score, dl_score, dl_label,phys_score, phys_feats):
        """
        Generate human-readable explanation for the classification
        
        Returns:
            str: Explanation text
        """
        explanations = []
        
        if final_score > 0.55:
            # AI GENERATED
            
            # Deep Learning contributions
            if dl_score > 0.55 and self.dl_ready:
                if "Fake" in dl_label or "Deepfake" in dl_label:
                    explanations.append(
                        f"Deep learning model detected synthetic voice patterns "
                        f"(confidence: {dl_score*100:.1f}%)"
                    )
            
            # Physics contributions
            if phys_score > 0.55:
                p_cv = phys_feats.get('pitch_cv', 0)
                i_std = phys_feats.get('intensity_std', 0)
                
                if i_std < 0.06:
                    explanations.append(
                        f"Unnaturally consistent energy levels detected "
                        f"(std: {i_std:.3f}, expected: >0.06)"
                    )
                
                if p_cv < 0.22 and p_cv > 0:
                    explanations.append(
                        f"Robotic pitch modulation patterns "
                        f"(CV: {p_cv:.2f}, expected: >0.22)"
                    )
                
                if not explanations or (i_std >= 0.06 and p_cv >= 0.22):
                    explanations.append(
                        "Acoustic parameters lack natural human variability"
                    )
            
            if not explanations:
                explanations.append(
                    "Voice exhibits characteristics consistent with AI generation"
                )
                
        else:
            # HUMAN
            explanations.append(
                "Voice exhibits natural acoustic variability and human speech characteristics"
            )
        
        return "; ".join(explanations)

    # ==========================================================
    # PART D: MAIN ANALYSIS FUNCTION
    # ==========================================================
    def analyze(self, audio_input, input_type="file"):
        """
        Main analysis function with configurable input types
        
        Args:
            audio_input: Either file path or base64 string
            input_type: "file" or "base64"
            
        Returns:
            dict: Analysis results following API response format
        """
        temp_file = None
        
        try:
            # Handle input type
            if input_type == "base64":
                temp_file = self.decode_base64_audio(audio_input)
                audio_path = temp_file
            elif input_type == "file":
                audio_path = audio_input
                if not os.path.exists(audio_path):
                    return {
                        "status": "error",
                        "error": f"Audio file not found: {audio_path}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Invalid input_type: {input_type}. Use 'file' or 'base64'"
                }

            print(f"🎵 Analyzing: {os.path.basename(audio_path)}")

            # 1. Detect language for Tamil-specific threshold
            language = self.detect_language(audio_path)
            threshold = self.tamil_ai_threshold if language == "Tamil" else self.default_ai_threshold
            if language == "Tamil":
                print(f"   📐 Using Tamil-specific AI threshold: {threshold}")

            # 2. Load audio once and split into chunks for max-pooling
            preloaded = self._load_audio_once(audio_path)
            y, sr, duration, was_truncated = preloaded
            chunks = self._chunk_audio(y, sr, duration)
            if not chunks:
                chunks = [preloaded]  # Very short audio: single "chunk" = full audio

            # 3. Run Physics and DL per chunk; take max score (if any chunk is AI → whole file AI)
            best_score = -1.0
            best_phys_score, best_phys_method, best_phys_feats = 0.0, "", {}
            best_dl_score, best_dl_label = 0.0, ""

            for idx, preloaded_chunk in enumerate(chunks):
                phys_score, phys_method, phys_feats = 0.0, "", {}
                dl_score, dl_label = 0.0, ""
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_phys = executor.submit(self.get_physics_score, None, preloaded_chunk)
                    future_dl = executor.submit(self.get_dl_score, None, preloaded_chunk)
                    for future in as_completed([future_phys, future_dl]):
                        try:
                            result = future.result()
                            if len(result) == 3:
                                phys_score, phys_method, phys_feats = result
                            else:
                                dl_score, dl_label = result
                        except Exception as e:
                            print(f"   ⚠️  Chunk {idx+1} task error: {e}")
                            if future is future_phys:
                                phys_score, phys_method, phys_feats = 0.0, f"Error: {e}", {'duration': 0, 'was_truncated': False}
                            else:
                                dl_score, dl_label = 0.0, f"DL Error: {e}"

                chunk_score = self.physics_weight * phys_score + self.dl_weight * dl_score
                if chunk_score > best_score:
                    best_score = chunk_score
                    best_phys_score, best_phys_method, best_phys_feats = phys_score, phys_method, phys_feats
                    best_dl_score, best_dl_label = dl_score, dl_label
                print(f"   Chunk {idx+1}/{len(chunks)} score: {chunk_score:.3f} (phys={phys_score:.2f}, dl={dl_score:.2f})")

            final_score = round(best_score, 2)
            phys_score, phys_feats = best_phys_score, best_phys_feats
            dl_score, dl_label = best_dl_score, best_dl_label

            # 4. Classification with language-specific threshold
            classification = "AI_GENERATED" if final_score > threshold else "HUMAN"

            # 5. Generate explanation (using best chunk's scores/feats)
            explanation = self.generate_explanation(
                final_score,
                dl_score,
                dl_label,
                phys_score,
                phys_feats
            )

            # 6. Return API-compliant response
            return {
                "status": "success",
                "classification": classification,
                "confidenceScore": float(final_score),
                "explanation": explanation,
                "debug": {
                    "physics_score": float(phys_score),
                    "dl_score": float(dl_score),
                    "dl_label": dl_label,
                    "physics_weight": f"{self.physics_weight*100:.0f}%",
                    "dl_weight": f"{self.dl_weight*100:.0f}%",
                    "audio_duration": float(phys_feats.get('duration', 0)),
                    "was_truncated": bool(phys_feats.get('was_truncated', False)),
                    "chunks_used": len(chunks),
                    "ai_threshold_used": float(threshold),
                    "language": language,
                    "physics_features": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                        for k, v in phys_feats.items()
                                        if k not in ['duration', 'was_truncated']}
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    # ==========================================================
    # UTILITY: Update Weights
    # ==========================================================
    def update_weights(self, physics_weight, dl_weight):
        """
        Update ensemble weights dynamically
        
        Args:
            physics_weight: New physics weight (0-1)
            dl_weight: New DL weight (0-1)
        """
        total = physics_weight + dl_weight
        self.physics_weight = physics_weight / total
        self.dl_weight = dl_weight / total
        
        print(f"⚙️  Weights updated:")
        print(f"   Physics: {self.physics_weight*100:.0f}%")
        print(f"   DL: {self.dl_weight*100:.0f}%")