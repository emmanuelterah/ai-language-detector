# English Accent Detection Tool

This tool analyzes video content to detect English accents and provides confidence scores for the classification. It's designed to help evaluate spoken English for hiring purposes.

## Features

- Accepts video URLs (YouTube, Loom, or direct MP4 links)
- Extracts audio from videos
- Transcribes speech to text
- Analyzes accent patterns
- Provides confidence scores
- Simple web interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter a video URL in the input field and click "Analyze Accent"

4. Wait for the analysis to complete. The results will show:
   - Transcribed text
   - Detected accent
   - Confidence score
   - Brief explanation

## Technical Details

The tool uses several key components:
- `yt-dlp` for video downloading
- `moviepy` for audio extraction
- `speech_recognition` for speech-to-text conversion
- A rule-based approach for accent detection (can be enhanced with ML models)

## Limitations

- Currently supports basic English accent detection (British, American, Australian)
- Confidence scores are based on linguistic patterns
- Requires clear audio for best results
- Processing time depends on video length and quality

## Future Improvements

- Integration with more sophisticated accent detection models
- Support for more English accents
- Enhanced confidence scoring
- Batch processing capabilities #   a i - l a n g u a g e - d e t e c t o r  
 