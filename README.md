# Description
This program is used to automatically create lyric files for your music collection to use on apps like Plex or Music Assistant.

## Setup

### 1. Install Python
Make sure you have Python 3.8 or newer installed on your system.

    https://www.python.org/downloads/

Verify installation:

    python --version

 
 
### 2. Install Required Python Packages
The script needs:

whisper (OpenAI’s Whisper ASR model)

torch (PyTorch for running Whisper)

tqdm (progress bar)

Install them with:

    pip install torch tqdm git+https://github.com/openai/whisper.git

Note:

For GPU support, install the correct PyTorch version from pytorch.org matching your CUDA version 

On CPU-only machines, the above will install CPU PyTorch.




### 3. Run the Script
When the script is run for the first time, it will download the whisper model.

## Troubleshooting

###1. Slow Download of Model
You can manually download the model from https://github.com/openai/whisper/blob/main/whisper/__init__.py 


Put the downloaded model in
    
    Windows: C:\Users\<username>\.cache\whisper\<model>
    Linux: /home/<username>/.cache/whisper/<model>

