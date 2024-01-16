# Transcriber-WebUI


# *Overview*
## *Transcriber-WebUI is a tool that extracts audio from YouTube videos and utilizes OpenAI's Whisper for speech-to-text (STT) conversion. Designed for user convenience, it is implemented as a Streamlit app.*

# *Installation Guide*
Follow these steps to install and set up the environment:

## *1. Prerequisites:*

- Python 3.x
- PyTorch
- Cuda 12.x
- OpenAI-Whisper

## *2. Setting up a Virtual Environment:*

- Create a virtual environment:
```bash
  python -m venv .transcriber-webui
```

Activate the virtual environment:
```bash
  .\.transcriber-webui\Scripts\activate
```

## *3. Installing Dependencies:*

- Install PyTorch and Cuda:
```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Install Whisper:
```bash
  pip install git+https://github.com/openai/whisper.git
```

## *4. Cloning the Repository and Installing Packages:*

- Clone the repository and install required packages:
```bash
  git clone https://github.com/hwonheo/transcriber-webui.git
  cd transcriber-webui
  pip install -r requirements.txt
```


## *5. Running the Streamlit App:*

- To run the Streamlit app, enter the following command:
```bash
  streamlit run youtube_transcriber_app.py 
```


