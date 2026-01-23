# Building a Raspberry Pi Voice Bot Interface for Pipecat

A guide for creating a hardware voice interface that connects to a Pipecat WebRTC server.

## Overview

This guide covers building a standalone voice assistant device using:
- **Raspberry Pi Zero 2 W** (or any Pi with WiFi)
- **Audio HAT** (e.g., WM8960-based WhisPlay board)
- **Pipecat server** with SmallWebRTC transport

The device connects to your Pipecat server via WebRTC, sending microphone audio and playing bot responses through speakers.

## Hardware Requirements

| Component | Purpose |
|-----------|---------|
| Raspberry Pi Zero 2 W | Main processor |
| Audio HAT with codec (WM8960) | Mic input + speaker output |
| Button | Trigger connection |
| LCD display (optional) | Status feedback |
| Speaker | Audio output |

## Software Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  PyAudio    │───▶│   aiortc     │───▶│  WebRTC    │ │
│  │  (Mic In)   │    │  (WebRTC)    │    │  Server    │ │
│  └─────────────┘    └──────────────┘    └────────────┘ │
│  ┌─────────────┐           │                           │
│  │  PyAudio    │◀──────────┘                           │
│  │  (Speaker)  │    Audio from bot                     │
│  └─────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

## Key Dependencies

```bash
pip install aiortc aiohttp numpy pyaudio
```

## Critical Implementation Details

### 1. Audio Chunk Size for WebRTC/Opus

WebRTC with Opus codec requires **20ms audio frames**:

```python
SAMPLE_RATE = 16000
CHUNK_SIZE = 320  # 20ms at 16kHz (16000 * 0.020 = 320)
```

Using wrong chunk sizes causes codec errors like:
```
Discarding 960 audio samples due to Opus codec requiring 480 samples
```

### 2. Half-Duplex Audio (Echo Prevention)

The bot will hear itself unless you implement echo cancellation. The simplest approach is **half-duplex**: mute the microphone while the bot is speaking.

```python
last_bot_activity_time = 0.0
MUTE_DURATION = 1.5  # seconds

# In playback - detect when bot is speaking
if audio_peak > THRESHOLD:
    last_bot_activity_time = time.time()

# In mic capture - mute if bot recently spoke
if (time.time() - last_bot_activity_time) < MUTE_DURATION:
    audio_data = np.zeros_like(audio_data)  # Send silence
```

### 3. WM8960 Codec Configuration

The WM8960 codec requires specific ALSA mixer settings:

```python
def force_unmute_mic():
    """Ensure microphone is unmuted in hardware mixer."""
    import subprocess
    controls = [
        ('Capture', 'cap'),      # Enable capture
        ('Capture', '63'),       # Max gain
        ('ADC PCM', '255'),      # ADC volume (0 = MUTE, 255 = max)
        ('L Input Mux', 'L2'),   # Select mic input
        ('R Input Mux', 'R2'),
    ]
    for control, value in controls:
        subprocess.run(
            ['amixer', '-D', 'hw:wm8960soundcard', 'sset', control, value],
            capture_output=True, check=False
        )
```

**Warning:** `ADC PCM` controls digital volume, NOT loopback. Setting it to `0` mutes the microphone entirely.

### 4. Stereo Output Requirement

Many audio codecs (including WM8960) require stereo output even for mono audio:

```python
# Open stream as stereo
stream = audio.open(
    format=pyaudio.paInt16,
    channels=2,  # Must be stereo
    rate=16000,
    output=True
)

# Convert mono to stereo before writing
stereo_data = np.repeat(mono_data[:, np.newaxis], 2, axis=1)
```

### 5. Buffer Sizes for Pi Zero

Pi Zero has limited CPU. Use larger buffers to prevent underruns:

```python
# Output buffer (playback)
frames_per_buffer=1920  # 120ms buffer

# Input buffer (capture)
frames_per_buffer=320   # 20ms (must match Opus frame size)
```

### 6. WebRTC Connection Flow

```python
# 1. Create peer connection
pc = RTCPeerConnection(configuration=RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
))

# 2. Add audio transceiver (sendrecv for bidirectional)
pc.addTransceiver(audio_track, direction="sendrecv")

# 3. Create data channel (required by SmallWebRTC)
data_channel = pc.createDataChannel("data", ordered=True)

# 4. Create and send offer
await pc.setLocalDescription(await pc.createOffer())
# POST offer to server's /api/offer endpoint

# 5. Set remote description from server's answer
await pc.setRemoteDescription(answer)

# 6. Handle incoming audio track
@pc.on("track")
def on_track(track):
    if track.kind == "audio":
        asyncio.create_task(play_audio_track(track))
```

### 7. Custom Audio Track for PyAudio

aiortc's MediaPlayer doesn't work well with ALSA. Create a custom track:

```python
class PyAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=320
        )
        self.timestamp = 0

    async def recv(self):
        data = self.stream.read(320, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Apply half-duplex muting here

        audio_data = audio_data.reshape((1, -1))
        frame = AudioFrame.from_ndarray(audio_data, format='s16', layout='mono')
        frame.sample_rate = 16000
        frame.pts = self.timestamp
        self.timestamp += 320
        return frame
```

## Systemd Service Configuration

For running as a service with proper audio access:

```ini
[Unit]
Description=Voice Bot
After=network.target sound.target

[Service]
Type=simple
ExecStart=/path/to/venv/bin/python -u /path/to/bot.py
WorkingDirectory=/path/to/project
User=your_user
Group=audio

# Priority settings for real-time audio
Nice=-10
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Audio environment
Environment=PYTHONUNBUFFERED=1
Environment=SDL_AUDIODRIVER=alsa
Environment=ALSA_CARD=wm8960soundcard
Environment=PULSE_SERVER=none

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Bot hears itself | No echo cancellation | Implement half-duplex muting |
| Mic not recording | ADC PCM set to 0 | Set ADC PCM to 255 |
| Opus codec errors | Wrong chunk size | Use 320 samples (20ms at 16kHz) |
| Audio underruns | Buffer too small | Increase frames_per_buffer |
| Slow/crackling audio | CPU overload | Reduce processing, increase buffers |
| Service mode slow | Missing audio config | Add ALSA environment variables |

## RTVI Message Handling

Pipecat sends transcriptions via the data channel:

```python
@data_channel.on("message")
def on_message(message):
    data = json.loads(message)
    if data.get("label") == "rtvi-ai":
        msg_type = data.get("type")
        text = data.get("data", {}).get("text")

        if msg_type == "user-transcription":
            print(f"User: {text}")
        elif msg_type == "bot-transcription":
            print(f"Bot: {text}")
```

## Testing Checklist

1. **Mic input works**: Run `arecord -D hw:wm8960soundcard -f S16_LE -r 16000 test.wav`
2. **Speaker output works**: Run `aplay -D hw:wm8960soundcard test.wav`
3. **Mixer settings correct**: Run `amixer -D hw:wm8960soundcard contents`
4. **WebRTC connects**: Check for "Connection state: connected" log
5. **Audio reaches server**: Check server logs for received audio frames
6. **Half-duplex works**: Bot should not transcribe its own speech

## Performance Tips for Pi Zero

1. **Avoid LCD updates during calls** - Drawing to screen blocks audio
2. **Use stride when checking audio levels** - `np.max(np.abs(audio[::10]))`
3. **Pre-allocate numpy arrays** - Avoid allocation in hot paths
4. **Use larger output buffers** - 1920 samples (120ms) prevents underruns
5. **Disable debug output** - Print statements slow down real-time audio

## Project Structure

```
project/
├── bot_half_duplex.py    # Main bot script
├── sync_to_rp.py         # Deploy script to Pi
├── whisplay.service      # Systemd service file
├── .env                  # API keys (not committed)
└── requirements.txt      # Dependencies
```

## References

- [Pipecat Documentation](https://github.com/pipecat-ai/pipecat)
- [aiortc Documentation](https://aiortc.readthedocs.io/)
- [WM8960 Datasheet](https://www.waveshare.com/wiki/WM8960_Audio_HAT)
