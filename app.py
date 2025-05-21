import streamlit as st
import yt_dlp
import os
import cv2
import speech_recognition as sr
import tempfile
import numpy as np
from collections import Counter
from pydub import AudioSegment
import subprocess
import shutil
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import nest_asyncio
from functools import lru_cache
from pytube import YouTube
import re

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Create a directory for downloads and analysis
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work')
os.makedirs(WORK_DIR, exist_ok=True)

# Global state for model
MODEL_STATE = {
    'classifier': None,
    'initialized': False
}

def initialize_model():
    """Initialize the model outside of Streamlit's context"""
    if not MODEL_STATE['initialized']:
        try:
            # Load model and tokenizer separately
            model_name = "facebook/bart-large-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline with explicit components
            MODEL_STATE['classifier'] = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            MODEL_STATE['initialized'] = True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            MODEL_STATE['initialized'] = False

# Initialize model at module level
initialize_model()

@st.cache_resource
def get_classifier():
    """Get the classifier from global state"""
    if not MODEL_STATE['initialized']:
        initialize_model()
    return MODEL_STATE['classifier']

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

def is_youtube_url(url):
    """Check if the URL is a YouTube URL"""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(youtube_regex, url))

def download_with_pytube(url):
    """Download video using pytube"""
    try:
        yt = YouTube(url)
        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            raise Exception("No audio stream found")
            
        # Download to a temporary file
        temp_path = os.path.join(WORK_DIR, f"audio_{int(np.random.random() * 1000000)}.mp4")
        audio_stream.download(output_path=WORK_DIR, filename=os.path.basename(temp_path))
        
        return temp_path
    except Exception as e:
        print(f"Pytube download failed: {str(e)}")
        return None

def download_video(url):
    """Download video from URL and return the local path"""
    # Clean up any existing files
    cleanup_work_dir()
    
    # Create a unique filename
    temp_path = os.path.join(WORK_DIR, f"audio_{int(np.random.random() * 1000000)}.webm")
    
    try:
        # Try pytube first for YouTube URLs
        if is_youtube_url(url):
            downloaded_path = download_with_pytube(url)
            if downloaded_path and os.path.exists(downloaded_path):
                return downloaded_path
        
        # Fallback to yt-dlp for non-YouTube URLs or if pytube fails
        ydl_opts = {
            'format': 'bestaudio[ext=webm]/bestaudio/best',
            'outtmpl': temp_path,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'force_generic_extractor': False,
            'verbose': True,
            'ignoreerrors': True,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'keepvideo': True,
            'writethumbnail': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'skip_download': False,
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'no_warnings': False,
            'quiet': False,
            'verbose': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, try to get video info
            info = ydl.extract_info(url, download=False)
            if not info:
                raise Exception("Could not extract video information")
            
            # Download the video
            ydl.download([url])
        
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

def analyze_accent(text):
    """Analyze accent using machine learning model"""
    if not text:
        return {
            'accent': 'Unknown',
            'confidence': 0,
            'explanation': 'No text provided for analysis'
        }
    
    # Get the classifier from global state
    classifier = get_classifier()
    if not classifier:
        return {
            'accent': 'Error',
            'confidence': 0,
            'explanation': 'Failed to load accent classifier'
        }
    
    # Define accent categories with their characteristics
    accent_categories = [
        "British English accent with formal vocabulary and traditional expressions",
        "American English accent with casual vocabulary and modern expressions",
        "Australian English accent with colloquial vocabulary and informal expressions",
        "Indian English accent with unique vocabulary and grammar patterns",
        "African English accent with distinctive vocabulary and speech patterns",
        "Caribbean English accent with unique vocabulary and rhythm"
    ]
    
    try:
        # Perform zero-shot classification
        result = classifier(
            text,
            accent_categories,
            multi_label=False
        )
        
        # Get the best match
        best_match_idx = result['labels'].index(result['labels'][0])
        confidence = result['scores'][0] * 100
        
        # Generate explanation based on the detected accent
        accent = result['labels'][0]
        explanation = f"Detected {accent} with {confidence:.1f}% confidence. "
        
        # Add specific observations based on the accent
        if "British" in accent:
            explanation += "The text shows characteristics of British English, including formal vocabulary and traditional expressions."
        elif "American" in accent:
            explanation += "The text shows characteristics of American English, including casual vocabulary and modern expressions."
        elif "Australian" in accent:
            explanation += "The text shows characteristics of Australian English, including colloquial vocabulary and informal expressions."
        elif "Indian" in accent:
            explanation += "The text shows characteristics of Indian English, including unique vocabulary and grammar patterns."
        elif "African" in accent:
            explanation += "The text shows characteristics of African English, including distinctive vocabulary and speech patterns."
        elif "Caribbean" in accent:
            explanation += "The text shows characteristics of Caribbean English, including unique vocabulary and rhythm."
        
        return {
            'accent': accent,
            'confidence': confidence,
            'explanation': explanation
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
                        result = analyze_accent(text)
                        
                        # Display results
                        st.success(f"Detected Accent: {result['accent']}")
                        st.info(f"Confidence: {result['confidence']:.1f}%")
                        st.write(result['explanation'])
                    
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