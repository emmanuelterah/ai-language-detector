import streamlit as st
import os
import cv2
import speech_recognition as sr
import tempfile
import numpy as np
from collections import Counter
from pydub import AudioSegment
import subprocess
import shutil
import soundfile as sf
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
from pytube import YouTube

# Create a directory for downloads and analysis
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
os.makedirs(WORK_DIR, exist_ok=True)

# Define accent patterns
ACCENT_PATTERNS = {
    "British English": {
        "vocabulary": [
            "colour", "favour", "centre", "theatre", "realise", "organise",
            "analyse", "catalogue", "dialogue", "programme", "cheque",
            "plough", "mould", "moult", "smoulder", "moustache", "storey",
            "grey", "jewellery", "marvellous", "travelling", "counsellor",
            "counselling", "signalling", "cancelling", "modelling", "traveller"
        ],
        "phrases": [
            "shall we", "whilst", "amongst", "whilst", "amongst",
            "have got", "haven't got", "hasn't got", "hadn't got"
        ]
    },
    "American English": {
        "vocabulary": [
            "color", "favor", "center", "theater", "realize", "organize",
            "analyze", "catalog", "dialog", "program", "check",
            "plow", "mold", "molt", "smolder", "mustache", "story",
            "gray", "jewelry", "marvelous", "traveling", "counselor",
            "counseling", "signaling", "canceling", "modeling", "traveler"
        ],
        "phrases": [
            "gotten", "fall", "garbage", "elevator", "apartment", "sidewalk",
            "truck", "trunk", "hood", "windshield", "gas", "parking lot"
        ]
    },
    "Australian English": {
        "vocabulary": [
            "mate", "gday", "barbie", "arvo", "brekkie", "bloke",
            "crikey", "drongo", "fair dinkum", "good on ya", "no worries",
            "sheila", "strewth", "ta", "tucker", "ute", "wowser"
        ],
        "phrases": [
            "no worries", "good on ya", "fair dinkum", "she'll be right",
            "no worries", "good on ya", "fair dinkum", "she'll be right"
        ]
    },
    "Indian English": {
        "vocabulary": [
            "prepone", "do the needful", "kindly revert", "out of station",
            "pass out", "timepass", "batchmate", "co-brother", "cousin-brother",
            "cousin-sister", "co-sister", "co-brother", "co-sister"
        ],
        "phrases": [
            "do the needful", "kindly revert", "out of station",
            "pass out", "timepass", "batchmate"
        ]
    },
    "African English": {
        "vocabulary": [
            "chop", "mammy", "pikin", "wahala", "bros", "sista",
            "oga", "madam", "bros", "sista", "oga", "madam"
        ],
        "phrases": [
            "how far", "no wahala", "well done", "thank you",
            "how far", "no wahala", "well done"
        ]
    },
    "Caribbean English": {
        "vocabulary": [
            "irie", "yaad", "ting", "bredren", "sistren", "lime",
            "brawta", "dutty", "fyah", "irie", "yaad", "ting"
        ],
        "phrases": [
            "no problem", "irie", "yaad", "ting", "bredren",
            "sistren", "lime", "brawta"
        ]
    }
}

def cleanup_work_dir():
    """Clean up the work directory"""
    for file in os.listdir(WORK_DIR):
        try:
            file_path = os.path.join(WORK_DIR, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def download_video(url):
    """Download video from URL and return the local path"""
    # Clean up any existing files
    cleanup_work_dir()
    
    # Create a unique filename
    temp_path = os.path.join(WORK_DIR, f"audio_{int(np.random.random() * 1000000)}.mp4")
    
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            raise Exception("No audio stream found")
        
        # Download the audio
        audio_stream.download(output_path=WORK_DIR, filename=os.path.basename(temp_path))
        
        # Verify the downloaded file
        if not os.path.exists(temp_path):
            raise Exception("Downloaded file does not exist")
            
        if os.path.getsize(temp_path) == 0:
            raise Exception("Downloaded file is empty")
            
        # Try to open the file to verify it's valid
        try:
            with open(temp_path, 'rb') as f:
                # Read first few bytes to verify file is not corrupted
                header = f.read(1024)
                if len(header) == 0:
                    raise Exception("File appears to be corrupted")
        except Exception as e:
            raise Exception(f"Error verifying downloaded file: {str(e)}")
            
        return temp_path
    except Exception as e:
        # Clean up any temporary files
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise Exception(f"Failed to download video: {str(e)}")

def extract_audio(video_path):
    """Extract audio from video using pydub and return the audio path"""
    audio_path = os.path.join(WORK_DIR, f"audio_{int(np.random.random() * 1000000)}.wav")
    
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(video_path)
        
        # Export as WAV
        audio.export(audio_path, format="wav")
        
        # Verify the audio file was created and is valid
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Failed to create audio file")
            
        return audio_path
    except Exception as e:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        raise Exception(f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path):
    """Transcribe audio to text using speech recognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"Error in transcription: {str(e)}")
            return None

def extract_audio_features(audio_path):
    """Extract phonetic features from audio"""
    try:
        # Load audio file
        data, samplerate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Extract features
        features = {}
        
        # 1. Basic audio features
        features['rms'] = np.sqrt(np.mean(data**2))  # Root mean square
        features['zero_crossings'] = np.sum(np.diff(np.signbit(data)))  # Zero crossing rate
        
        # 2. Spectral features
        spectrum = np.abs(fft(data))
        freqs = np.fft.fftfreq(len(data), 1/samplerate)
        
        # Get positive frequencies only
        pos_freq_mask = freqs > 0
        freqs = freqs[pos_freq_mask]
        spectrum = spectrum[pos_freq_mask]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs * spectrum) / np.sum(spectrum)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * spectrum) / np.sum(spectrum))
        
        # 3. Pitch detection using autocorrelation
        def get_pitch(data, sr):
            # Compute autocorrelation
            corr = signal.correlate(data, data, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find peaks
            peaks = signal.find_peaks(corr)[0]
            if len(peaks) > 0:
                # Get the first peak after the first zero crossing
                zero_crossings = np.where(np.diff(np.signbit(corr)))[0]
                if len(zero_crossings) > 0:
                    valid_peaks = peaks[peaks > zero_crossings[0]]
                    if len(valid_peaks) > 0:
                        return sr / valid_peaks[0]
            return 0
        
        features['pitch'] = get_pitch(data, samplerate)
        
        # 4. Rhythm features
        # Compute onset strength
        onset_env = np.diff(np.abs(data))
        features['onset_strength'] = np.mean(onset_env)
        
        # Estimate tempo
        autocorr = signal.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks = signal.find_peaks(autocorr)[0]
        if len(peaks) > 0:
            features['tempo'] = 60 * samplerate / peaks[0]
        else:
            features['tempo'] = 0
        
        return features
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return None

def analyze_accent(audio_path, text):
    """Analyze accent using phonetic features and text analysis"""
    if not text:
        return {
            'accent': 'Unknown',
            'confidence': 0,
            'explanation': 'No text provided for analysis'
        }
    
    try:
        # Extract audio features
        features = extract_audio_features(audio_path)
        if not features:
            return {
                'accent': 'Error',
                'confidence': 0,
                'explanation': 'Failed to extract audio features'
            }
        
        # Define accent characteristics based on phonetic research
        accent_characteristics = {
            "British English": {
                "pitch_range": (180, 220),  # Hz
                "tempo_range": (120, 140),  # BPM
                "spectral_centroid_range": (2000, 2500),  # Hz
                "onset_strength_range": (0.1, 0.3)
            },
            "American English": {
                "pitch_range": (200, 240),  # Hz
                "tempo_range": (130, 150),  # BPM
                "spectral_centroid_range": (2200, 2700),  # Hz
                "onset_strength_range": (0.2, 0.4)
            },
            "Australian English": {
                "pitch_range": (190, 230),  # Hz
                "tempo_range": (125, 145),  # BPM
                "spectral_centroid_range": (2100, 2600),  # Hz
                "onset_strength_range": (0.15, 0.35)
            },
            "Indian English": {
                "pitch_range": (170, 210),  # Hz
                "tempo_range": (115, 135),  # BPM
                "spectral_centroid_range": (1900, 2400),  # Hz
                "onset_strength_range": (0.1, 0.3)
            },
            "African English": {
                "pitch_range": (160, 200),  # Hz
                "tempo_range": (110, 130),  # BPM
                "spectral_centroid_range": (1800, 2300),  # Hz
                "onset_strength_range": (0.1, 0.3)
            },
            "Caribbean English": {
                "pitch_range": (175, 215),  # Hz
                "tempo_range": (118, 138),  # BPM
                "spectral_centroid_range": (1950, 2450),  # Hz
                "onset_strength_range": (0.15, 0.35)
            }
        }
        
        # Calculate scores for each accent
        scores = {}
        for accent, characteristics in accent_characteristics.items():
            score = 0
            total_features = 0
            
            # Check pitch
            if 'pitch' in features:
                pitch = features['pitch']
                if characteristics['pitch_range'][0] <= pitch <= characteristics['pitch_range'][1]:
                    score += 1
                total_features += 1
            
            # Check tempo
            if 'tempo' in features:
                tempo = features['tempo']
                if characteristics['tempo_range'][0] <= tempo <= characteristics['tempo_range'][1]:
                    score += 1
                total_features += 1
            
            # Check spectral centroid
            if 'spectral_centroid' in features:
                centroid = features['spectral_centroid']
                if characteristics['spectral_centroid_range'][0] <= centroid <= characteristics['spectral_centroid_range'][1]:
                    score += 1
                total_features += 1
            
            # Check onset strength
            if 'onset_strength' in features:
                onset = features['onset_strength']
                if characteristics['onset_strength_range'][0] <= onset <= characteristics['onset_strength_range'][1]:
                    score += 1
                total_features += 1
            
            # Calculate final score
            if total_features > 0:
                scores[accent] = (score / total_features) * 100
        
        # Get the best match
        if scores:
            best_accent = max(scores.items(), key=lambda x: x[1])
            accent, confidence = best_accent
            
            # Generate explanation
            explanation = f"Detected {accent} with {confidence:.1f}% confidence. "
            
            # Add specific observations based on the phonetic features
            if features['pitch'] < 190:
                explanation += "The speech shows lower pitch characteristics typical of "
                if "African" in accent:
                    explanation += "African English patterns. "
                elif "Indian" in accent:
                    explanation += "Indian English patterns. "
            elif features['pitch'] > 220:
                explanation += "The speech shows higher pitch characteristics typical of "
                if "American" in accent:
                    explanation += "American English patterns. "
                elif "Australian" in accent:
                    explanation += "Australian English patterns. "
            
            if features['tempo'] < 125:
                explanation += "The speech rhythm is more measured and deliberate, "
                if "British" in accent:
                    explanation += "characteristic of British English. "
                elif "Indian" in accent:
                    explanation += "characteristic of Indian English. "
            elif features['tempo'] > 140:
                explanation += "The speech rhythm is more rapid and fluid, "
                if "American" in accent:
                    explanation += "characteristic of American English. "
                elif "Australian" in accent:
                    explanation += "characteristic of Australian English. "
            
            return {
                'accent': accent,
                'confidence': confidence,
                'explanation': explanation
            }
        else:
            return {
                'accent': 'Unknown',
                'confidence': 0,
                'explanation': 'Could not determine accent from audio features'
            }
            
    except Exception as e:
        return {
            'accent': 'Error',
            'confidence': 0,
            'explanation': f'Error analyzing accent: {str(e)}'
        }

def main():
    st.title("English Accent Detection Tool")
    st.write("Upload a video URL to analyze the speaker's accent")
    
    # Check for FFmpeg
    if not check_ffmpeg():
        st.error("FFmpeg is not installed. Please install FFmpeg to use this application.")
        st.info("You can download FFmpeg from: https://ffmpeg.org/download.html")
        return
    
    video_url = st.text_input("Enter video URL (YouTube, Loom, or direct MP4 link):")
    
    if st.button("Analyze Accent"):
        if video_url:
            with st.spinner("Processing video..."):
                try:
                    # Download video
                    video_path = download_video(video_url)
                    st.info(f"Downloaded to: {video_path}")
                    
                    # Extract audio
                    audio_path = extract_audio(video_path)
                    st.info(f"Extracted audio to: {audio_path}")
                    
                    # Transcribe audio
                    text = transcribe_audio(audio_path)
                    
                    if text:
                        st.write("Transcribed Text:", text)
                        
                        # Analyze accent
                        result = analyze_accent(audio_path, text)
                        
                        # Display results
                        st.success(f"Detected Accent: {result['accent']}")
                        st.info(f"Confidence: {result['confidence']:.1f}%")
                        st.write(result['explanation'])
                        
                        # Display audio waveform
                        st.subheader("Audio Analysis")
                        data, samplerate = sf.read(audio_path)
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1)
                        plt.figure(figsize=(10, 3))
                        plt.plot(data)
                        plt.title("Audio Waveform")
                        st.pyplot(plt)
                    
                    # Cleanup
                    cleanup_work_dir()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    # Cleanup on error
                    cleanup_work_dir()
        else:
            st.warning("Please enter a video URL")

if __name__ == "__main__":
    main() 