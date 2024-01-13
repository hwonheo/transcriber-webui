import streamlit as st
import os
import pandas as pd
from pytube import YouTube
import torch
import whisper
import ctranslate2
import kss
from urllib.parse import urlparse, parse_qs
import csv
import json

# Define Whisper model sizes and output formats
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "tsv", "json"]

# YouTubeTranscriber class definition
class YouTubeTranscriber:
    """A class to transcribe YouTube audio using Whisper."""

    def __init__(self, url, model_size='large-v3', temperature=0, output_format="txt", language=None):
        self.url = url
        self.model_size = model_size
        self.temperature = temperature
        self.output_format = output_format
        self.language = language
        self.video_id = None

    def download_audio(self):
        """Downloads audio from a YouTube URL."""
        if not self.url.startswith('http'):
            self.url = f"https://www.youtube.com/watch?v={self.url}"
        yt = YouTube(self.url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download()
        base, _ = os.path.splitext(audio_file)
        audio_file_mp3 = base + '.mp3'
        if os.path.exists(audio_file_mp3):
            os.remove(audio_file_mp3)
        os.rename(audio_file, audio_file_mp3)
        self.video_id = yt.video_id
        return audio_file_mp3

    def transcribe_audio(self, audio_file):
        """Transcribes audio using Whisper and splits sentences with kss."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(self.model_size)
        model.to(self.device)
        options = {"task": "transcribe"}
        if self.language:
            options["language"] = self.language
        result = model.transcribe(audio_file, temperature=self.temperature, **options)
        text = result['text']
        sentences = kss.split_sentences(text)
        return sentences, result

    def save_transcription(self, sentences, output_dir):
        """Saves the transcribed text in the specified output format."""
        output_path = os.path.join(output_dir, f"whisper_dialog_{self.video_id}.{self.output_format}")

        if self.output_format == 'tsv':
            with open(output_path, "w", newline='', encoding='utf-8') as file:
                tsv_writer = csv.writer(file, delimiter='\t')
                for sentence in sentences:
                    tsv_writer.writerow([sentence])
        elif self.output_format == 'json':
            with open(output_path, "w", encoding='utf-8') as file:
                json.dump(sentences, file, ensure_ascii=False, indent=4)
        else:
            with open(output_path, "w", encoding='utf-8') as file:
                file.write('\n\n'.join(sentences))

        return output_path

# Additional functions
def create_output_directory(output_dir):
    """Creates an output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def extract_urls_from_file(uploaded_file):
    """Extracts URLs from an Excel, CSV, or text file."""
    file_name = uploaded_file.name
    if file_name.lower().endswith(('.xlsx', '.xls', '.csv')):
        return extract_urls_from_excel_or_csv(uploaded_file)
    else:
        return extract_urls_from_text_file(uploaded_file)

def extract_urls_from_excel_or_csv(uploaded_file):
    """Extracts URLs from an Excel or CSV file."""
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        urls = []
        for col in df.columns:
            if df[col].dtype == 'object' and any(df[col].str.contains('http://|https://|bit.ly', regex=True)):
                urls.extend(df[col].dropna().tolist())
        return urls
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def extract_urls_from_text_file(uploaded_file):
    """Extracts URLs from a text file."""
    with uploaded_file as file:
        return [line.strip() for line in file if urlparse(line.strip()).scheme in ['http', 'https', 'bit.ly']]

def validate_youtube_url(url):
    """Validates if the URL is a YouTube URL."""
    try:
        YouTube(url)
        return True
    except:
        return False

def transcribe_and_save(url, model_size, temperature, output_format, output_dir, language):
    """Transcribes a YouTube video and saves the transcription."""
    try:
        transcriber = YouTubeTranscriber(url, model_size, temperature, output_format, language)
        audio_file = transcriber.download_audio()
        transcribed_text, result = transcriber.transcribe_audio(audio_file)

        create_output_directory(output_dir)
        output_file_path = transcriber.save_transcription(transcribed_text, output_dir)
        return output_file_path
    except Exception as e:
        st.error(f"Error processing URL {url}: {e}")
        return None

def extract_video_id_from_url(url):
    """Extracts the video ID from a YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    return None

def get_youtube_title(url):
    """Gets the title of a YouTube video."""
    try:
        yt = YouTube(url)
        return yt.title
    except Exception as e:
        st.error(f"Error retrieving YouTube video title: {e}")
        return None

# Streamlit app main function
def main():
    st.title("YouTube Transcriber")
    st.subheader('_Transcribe YouTube Scrpts simply using Whisper Large-v3_', divider='rainbow')

    # Sidebar settings
    with st.sidebar:
        st.header('_Transcriber Settings_', divider='rainbow')
        st.markdown("\n")
        model_size = st.sidebar.selectbox("Choose Whisper Model Size", WHISPER_MODEL_SIZES, index=6)
        st.markdown("\n")
        language = st.sidebar.selectbox("Choose Language", ["ko", "en", "jp", "cn", None], index=4)
        st.markdown("\n")
        temperature = st.sidebar.slider("Set Temperature", 0.0, 1.0, 0.0)
        st.markdown("\n")
        output_format = st.sidebar.selectbox("Output Format", OUTPUT_FORMATS)
        st.markdown("\n")
        output_dir = st.sidebar.text_input("Output Directory", value="output")

    # Main page
    url = st.text_input("Enter YouTube URL")
    if url:
        video_id = extract_video_id_from_url(url)
        video_title = get_youtube_title(url)
        if video_id and video_title:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            st.image(thumbnail_url, caption=video_title)

    uploaded_file = st.file_uploader("Or Upload File (Excel/CSV/Text)", type=["xlsx", "xls", "csv", "txt"])

    if st.button("Transcribe YouTube Video"):
        urls = []
        if uploaded_file:
            urls = extract_urls_from_file(uploaded_file)
        
        if url:
            urls.append(url)

        for url in urls:
            if not validate_youtube_url(url):
                st.error(f"Invalid YouTube URL: {url}")
                continue

            output_file_path = transcribe_and_save(url, model_size, temperature, output_format, output_dir, language)
            if output_file_path:
                st.success(f"Transcription saved to {output_file_path}")
                with open(output_file_path, "rb") as file:
                    st.download_button("Download Transcription", file, file_name=os.path.basename(output_file_path))

if __name__ == "__main__":
    main()
