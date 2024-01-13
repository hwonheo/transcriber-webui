import os
import argparse
import pandas as pd
import csv
import json
from urllib.parse import urlparse
from pytube import YouTube
import torch
import whisper
import ctranslate2
import kss

# Define Whisper model sizes and output formats
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "tsv", "json"]

class YouTubeTranscriber:
    """A class to transcribe YouTube audio using Whisper."""

    def __init__(self, url, model_size='medium', temperature=0, output_format="txt", language=None):
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

def create_output_directory(output_dir):
    """Creates an output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_arguments():
    """Parses and returns command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe YouTube video audio to text.")
    parser.add_argument("--input", help="Input file path (Excel, CSV, or text file containing URLs)", required=False)
    parser.add_argument("--url", help="YouTube video URL or ID", required=False)
    parser.add_argument("--model_size", default="medium", choices=WHISPER_MODEL_SIZES, help="Whisper model size to use")
    parser.add_argument("--language", choices=["ko", "en", "jp", "cn"], default=None,
                        help="Language spoken in the audio (ko, en, jp, cn) or None for automatic detection")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--output_format", default="txt", choices=OUTPUT_FORMATS, help="Format of the output file")
    parser.add_argument("--output", default="output", help="Output directory to save the transcription")

    args = parser.parse_args()
    return args

def extract_urls_from_file(file_path):
    """Extracts URLs from an Excel, CSV, or text file."""
    if file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
        return extract_urls_from_excel_or_csv(file_path)
    else: 
        return extract_urls_from_text_file(file_path)

def extract_urls_from_excel_or_csv(file_path):
    """Extracts URLs from an Excel or CSV file."""
    try:
        df = pd.read_csv(file_path) if file_path.lower().endswith('.csv') else pd.read_excel(file_path)
        urls = []
        for col in df.columns:
            if df[col].dtype == 'object' and any(df[col].str.contains('http://|https://|bit.ly', regex=True)):
                urls.extend(df[col].dropna().tolist())
        return urls
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def extract_urls_from_text_file(file_path):
    """Extracts URLs from a text file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if urlparse(line.strip()).scheme in ['http', 'https', 'bit.ly']]

def validate_youtube_url(url):
    """Validates if the URL is a YouTube URL."""
    try:
        YouTube(url)
        return True
    except:
        return False

def transcribe_and_save(url, args):
    """Transcribes a YouTube video and saves the transcription."""
    try:
        transcriber = YouTubeTranscriber(url, args.model_size, args.temperature, args.output_format, args.language)
        audio_file = transcriber.download_audio()
        transcribed_text, result = transcriber.transcribe_audio(audio_file)

        create_output_directory(args.output)
        output_file_path = transcriber.save_transcription(transcribed_text, args.output)
        print(f"Transcription saved to {output_file_path}")

        os.remove(audio_file)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")

def main():
    """Main function to execute the script."""
    args = get_arguments()
    urls = []

    if args.input:
        urls = extract_urls_from_file(args.input)
        if not urls:
            print("No URLs in files..")
            return

    if args.url:
        urls.append(args.url)

    if not urls:
        print("No URL provided.")
        return

    for url in urls:
        if not validate_youtube_url(url):
            print(f"Invalid YouTube URL: {url}")
            continue
        transcribe_and_save(url, args)

if __name__ == "__main__":
    main()
