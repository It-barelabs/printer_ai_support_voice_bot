# Printer AI Support Voice Bot

A real-time voice AI support bot built with [Pipecat](https://github.com/pipecat-ai/pipecat), FastAPI, and WebRTC.

## Features

- **Real-time Voice Interaction:** Low-latency voice conversation using WebRTC.
- **AI Pipeline:**
  - **Speech-to-Text:** Deepgram
  - **LLM:** Google Vertex AI
  - **Text-to-Speech:** ElevenLabs
- **Web Interface:** Simple frontend for connecting to the bot.

## Prerequisites

- Python 3.12+
- API Keys for:
  - Deepgram
  - ElevenLabs
  - Google Cloud (Vertex AI)

## Installation

1. Clone the repository.
2. Install dependencies using `uv` (recommended):
   ```bash
   uv sync
   ```
   Or using pip:
   ```bash
   pip install .
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
# Ensure Google Cloud credentials are set up for Vertex AI
```

## Usage

Start the server:

```bash
python server.py
```

The application will run on `http://localhost:7860`. Open this URL in your browser to interact with the bot.

