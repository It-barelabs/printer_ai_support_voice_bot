#!/usr/bin/env python3
"""
Raspberry Pi Bot Script
Connects to WebRTC server for bot interaction when button is pressed.

Requirements:
    - aiortc: pip install aiortc
    - aiohttp: pip install aiohttp
    - WhisPlay board driver (from ../Driver directory)
    - Raspberry Pi with WhisPlay board
    - Server running at http://192.168.1.213:7860 (or specify with --server)

Usage:
    python rp/bot.py [--server URL] [--audio-input DEVICE]

Features:
    - Button press triggers WebRTC connection to bot server
    - Visual feedback via LCD display and RGB LED
    - Audio input from microphone
    - Audio output from bot (may need additional setup)
    - RTVI message handling for transcriptions
    - Data channel support (required for SmallWebRTC)

Note: Audio playback may require additional configuration depending on your
Raspberry Pi audio setup. The script receives audio but playback implementation
may need to be extended based on your hardware configuration.
"""

import sys
import os
import asyncio
import argparse
import json
from time import sleep
from typing import Optional
from collections import deque
from PIL import Image

# Add path for WhisPlay board
sys.path.append(os.path.abspath("../Driver"))
from WhisPlay import WhisPlayBoard

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame
    import numpy as np
    import aiohttp
except ImportError as e:
    print(f"ERROR: Missing required dependencies. Please install:")
    print(f"  pip install aiortc aiohttp numpy")
    print(f"Error: {e}")
    sys.exit(1)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ pyaudio not available - will try alternative methods")

# Initialize board
board = WhisPlayBoard()
board.set_backlight(50)

# Global state
peer_connection: Optional[RTCPeerConnection] = None
pc_id: Optional[str] = None
server_url: str = "http://192.168.1.213:7860"
audio_input_device: Optional[str] = None
relay = MediaRelay()
logo_image_data = None  # Store logo image data
last_bot_activity_time = 0.0 # Timestamp of last audio packet from bot
global_audio_track = None # Reference to the active audio track for flushing
import time # Ensure time is available

# LCD Debug Display Mode
# Set to True to show printouts on LCD screen, False to show logo image
SHOW_DEBUG_ON_SCREEN = False
screen_log_buffer = []  # Buffer to store recent log lines
MAX_LOG_LINES = 10  # Max lines to show on screen
last_lcd_update = 0.0  # Throttle LCD updates
LCD_UPDATE_INTERVAL = 2.0  # Minimum seconds between LCD updates
screen_buffer_dirty = False  # Flag to track if buffer changed
lcd_paused = False  # Pause LCD updates during active call
import threading


def render_text_to_rgb565(lines, screen_width, screen_height):
    """Render text lines to RGB565 format for LCD display."""
    from PIL import ImageDraw, ImageFont

    # Create black background image
    img = Image.new('RGB', (screen_width, screen_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to use a small monospace font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 10)
        except:
            font = ImageFont.load_default()

    # Draw each line
    y_offset = 2
    line_height = 12
    for line in lines[-MAX_LOG_LINES:]:  # Show last N lines
        # Truncate long lines
        max_chars = screen_width // 6  # Approximate char width
        if len(line) > max_chars:
            line = line[:max_chars-2] + ".."
        # Color code based on content
        if "ERROR" in line or "❌" in line:
            color = (255, 100, 100)  # Red
        elif "✅" in line:
            color = (100, 255, 100)  # Green
        elif "⚠️" in line:
            color = (255, 255, 100)  # Yellow
        elif "🎤" in line or "MIC" in line:
            color = (100, 200, 255)  # Cyan
        elif "🤖" in line or "Bot:" in line:
            color = (200, 150, 255)  # Purple
        elif "👤" in line or "User:" in line:
            color = (100, 255, 200)  # Teal
        else:
            color = (200, 200, 200)  # Light gray
        draw.text((2, y_offset), line, font=font, fill=color)
        y_offset += line_height
        if y_offset > screen_height - line_height:
            break

    # Convert to RGB565
    pixel_data = []
    for y in range(screen_height):
        for x in range(screen_width):
            r, g, b = img.getpixel((x, y))
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            pixel_data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])

    return pixel_data


def refresh_debug_screen():
    """Refresh the debug screen (only when not in active call)."""
    global last_lcd_update, screen_buffer_dirty

    if not SHOW_DEBUG_ON_SCREEN or not screen_buffer_dirty or lcd_paused:
        return

    now = time.time()
    if (now - last_lcd_update) < LCD_UPDATE_INTERVAL:
        return  # Throttle updates

    try:
        pixel_data = render_text_to_rgb565(
            screen_log_buffer,
            board.LCD_WIDTH,
            board.LCD_HEIGHT
        )
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, pixel_data)
        last_lcd_update = now
        screen_buffer_dirty = False
    except Exception:
        pass


def screen_print(message, end="\n", force_refresh=False):
    """Print to console and buffer for LCD screen."""
    global screen_log_buffer, screen_buffer_dirty

    # Always print to console
    print(message, end=end, flush=True)

    # If debug mode enabled, buffer message for LCD
    if SHOW_DEBUG_ON_SCREEN:
        # Clean message for display (remove carriage return control)
        clean_msg = message.replace("\r", "").strip()
        if clean_msg:
            # Handle multi-line messages
            for line in clean_msg.split("\n"):
                if line.strip():
                    screen_log_buffer.append(line.strip())
            # Keep buffer size limited
            screen_log_buffer = screen_log_buffer[-MAX_LOG_LINES:]
            screen_buffer_dirty = True

            # Only refresh on important messages or forced
            if force_refresh or "User:" in message or "Bot:" in message or "Button" in message or "Connect" in message:
                refresh_debug_screen()


def load_jpg_as_rgb565(filepath, screen_width, screen_height):
    """Load and convert image to RGB565 format for display (from test.py)."""
    img = Image.open(filepath).convert('RGB')
    original_width, original_height = img.size

    aspect_ratio = original_width / original_height
    screen_aspect_ratio = screen_width / screen_height

    if aspect_ratio > screen_aspect_ratio:
        # Original image is wider, scale based on screen height
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        # Calculate horizontal offset to center the image
        offset_x = (new_width - screen_width) // 2
        # Crop the image to fit screen width
        cropped_img = resized_img.crop(
            (offset_x, 0, offset_x + screen_width, screen_height))
    else:
        # Original image is taller or has the same aspect ratio, scale based on screen width
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        # Calculate vertical offset to center the image
        offset_y = (new_height - screen_height) // 2
        # Crop the image to fit screen height
        cropped_img = resized_img.crop(
            (0, offset_y, screen_width, offset_y + screen_height))

    pixel_data = []
    for y in range(screen_height):
        for x in range(screen_width):
            r, g, b = cropped_img.getpixel((x, y))
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            pixel_data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])

    return pixel_data


def display_logo():
    """Display the logo on the screen (skipped if debug mode is on)."""
    global logo_image_data
    # Don't show logo if debug screen mode is enabled
    if SHOW_DEBUG_ON_SCREEN:
        refresh_debug_screen()
        return
    if logo_image_data is not None:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, logo_image_data)


def check_audio_systems(quiet=False):
    """Check what audio systems are available."""
    import subprocess
    systems = {
        'alsa': False,
        'pulseaudio': False,
    }
    
    # Check ALSA
    try:
        result = subprocess.run(
            ['arecord', '--version'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            systems['alsa'] = True
            if not quiet:
                print("✅ ALSA is available")
        else:
            if not quiet:
                print("⚠️ ALSA tools found but may not be working")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        if not quiet:
            print("❌ ALSA not found or not accessible")
    
    # Check PulseAudio
    try:
        result = subprocess.run(
            ['pulseaudio', '--check', '-v'],
            capture_output=True,
            text=True,
            timeout=2
        )
        systems['pulseaudio'] = True
        if not quiet:
            print("✅ PulseAudio is available")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        try:
            # Try checking if pulseaudio is running
            result = subprocess.run(
                ['pactl', 'info'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                systems['pulseaudio'] = True
                if not quiet:
                    print("✅ PulseAudio is available (via pactl)")
        except:
            if not quiet:
                print("❌ PulseAudio not found or not accessible")
    
    return systems


def list_audio_devices():
    """List available audio devices."""
    import subprocess
    devices = []
    
    print("\n📋 Checking audio systems...")
    systems = check_audio_systems()
    
    # List ALSA devices
    if systems['alsa']:
        try:
            result = subprocess.run(
                ['arecord', '-l'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            print("\n📋 ALSA audio input devices:")
            print(result.stdout)
            
            # Try to extract device names
            import re
            for line in result.stdout.split('\n'):
                if 'card' in line.lower():
                    match = re.search(r'card (\d+)', line)
                    if match:
                        card_num = match.group(1)
                        devices.append(f"hw:{card_num},0")
                        devices.append(f"plughw:{card_num},0")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️ Failed to list ALSA devices: {e}")
    
    # List PulseAudio devices
    if systems['pulseaudio']:
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sources'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            print("\n📋 PulseAudio audio input devices:")
            print(result.stdout)
            # PulseAudio devices are typically accessed via "default" or device name
            devices.append("pulse")
            devices.append("default")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️ Failed to list PulseAudio devices: {e}")
    
    return devices


def set_wm8960_volume_stable(volume_level: str):
    """
    Sets the 'Speaker' volume for the wm8960 sound card using the amixer command.
    """
    import subprocess
    CARD_NAME = 'wm8960soundcard'
    CONTROL_NAME = 'Speaker'
    DEVICE_ARG = f'hw:{CARD_NAME}'

    command = [
        'amixer',
        '-D', DEVICE_ARG,
        'sset',
        CONTROL_NAME,
        volume_level
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"INFO: Successfully set '{CONTROL_NAME}' volume to {volume_level} on card '{CARD_NAME}'.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to execute amixer.", file=sys.stderr)
        print(f"Command: {' '.join(command)}", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
    except FileNotFoundError:
        print("ERROR: 'amixer' command not found. Ensure it is installed and in PATH.", file=sys.stderr)

async def play_audio_track(track):
    global last_bot_activity_time
    if not PYAUDIO_AVAILABLE:
        return

    # Using 16000Hz as requested
    TARGET_RATE = 16000
    CARD_NAME = 'wm8960soundcard'
    audio = pyaudio.PyAudio()

    # Find device index
    device_index = None

    # Cache device count to avoid repeated calls
    device_count = audio.get_device_count()

    for i in range(device_count):
        try:
            info = audio.get_device_info_by_index(i)
            if CARD_NAME.lower() in info['name'].lower() and info['maxOutputChannels'] > 0:
                device_index = i
                break
        except Exception:
            continue

    # 1. Open stream at 16kHz STEREO
    # The WM8960 almost always requires 2 channels at the driver level
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=2, # Still use 2 channels
            rate=TARGET_RATE,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1920 # 120ms output buffer (larger to prevent underruns on Pi Zero)
        )
    except Exception as e:
        print(f"❌ Could not open 16kHz stream: {e}")
        return

    # is_speaking = True  <-- Removed static assignment, using volume detection
    silence_frames = 0
    THRESHOLD = 2500  # Adjusted: Ignore background hiss, but catch speech
    SILENCE_LIMIT = 75 # ~1.5 seconds hold time

    # Pre-allocate array for mute to avoid allocation in loop
    mute_array = None

    try:
        while True:
            try:
                frame = await track.recv()
            except Exception:
                # Track closed during disconnect
                break
            if frame is None: break

            # 2. Get Mono data (16-bit)
            array = frame.to_ndarray()
            mono_data = array[0] if len(array.shape) > 1 else array
            if mono_data.dtype == np.float32:
                mono_data = (mono_data * 32767).astype(np.int16)

            # Check volume level to toggle mute
            # OPTIMIZATION for Pi Zero: Only check every 10th sample to save CPU
            if len(mono_data) > 0:
                # Fast peak detection (stride 10)
                peak = np.max(np.abs(mono_data[::10]))
            else:
                peak = 0

            if peak > THRESHOLD:
                last_bot_activity_time = time.time()

            # 3. Handle incoming vs target rate
            # If the bot sends 48k or 24k, we downsample to 16k
            if frame.sample_rate != TARGET_RATE:
                num_samples = len(mono_data)
                new_len = int(num_samples * TARGET_RATE / frame.sample_rate)
                mono_data = np.interp(
                    np.linspace(0, num_samples, new_len),
                    np.arange(num_samples),
                    mono_data
                ).astype(np.int16)

            # 4. CONVERT TO STEREO (The essential step for WM8960)
            # If you send mono to this driver, it plays at half-speed (slow)
            stereo_data = np.repeat(mono_data[:, np.newaxis], 2, axis=1)

            # 5. Write to hardware
            try:
                stream.write(mono_data.tobytes(), exception_on_underflow=False)
            except OSError:
                # Stream closed during disconnect
                break

    finally:
        # Clean up audio resources
        try:
            if stream: stream.close()
        except Exception:
            pass
        try:
            audio.terminate()
        except Exception:
            pass
 




async def send_ice_candidate(candidate_dict: dict, pc_id: str, session: aiohttp.ClientSession):
    """Send ICE candidate to server."""
    try:
        async with session.patch(
            f"{server_url}/api/offer",
            json={
                "pc_id": pc_id,
                "candidates": [candidate_dict]
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"ERROR: Failed to send ICE candidate: {response.status} - {error_text}")
    except Exception as e:
        print(f"ERROR: Exception sending ICE candidate: {e}")


class PyAudioTrack(MediaStreamTrack):
    """Custom audio track using PyAudio (similar to how test.py uses pygame)."""
    kind = "audio"

    def __init__(self):
        super().__init__()
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("pyaudio is required for audio input")

        # Use same device approach as test.py - try wm8960soundcard first
        CARD_NAME = 'wm8960soundcard'

        # Create a separate PyAudio instance for input to avoid conflicts with output
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 320  # 20ms chunks at 16kHz (standard for Opus/WebRTC)
        self.timestamp = 0  # Track presentation timestamp
        self._stream_active = True  # Track if stream is active

        # Try to find the correct input device
        self.device_index = None # Store as instance variable for recovery
        device_count = self.audio.get_device_count()

        # First, try to find wm8960soundcard by name (Priority!)
        print("🔍 Scanning for wm8960soundcard...")
        for i in range(device_count):
            try:
                info = self.audio.get_device_info_by_index(i)
                if CARD_NAME.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
                    self.device_index = i
                    print(f"✅ Found audio input device: {info['name']} (index {i})")
                    break
            except:
                continue

        # If not found, use default input device
        if self.device_index is None:
            try:
                self.device_index = self.audio.get_default_input_device_info()['index']
                print(f"✅ Using default audio input device (index {self.device_index})")
            except:
                print("⚠️ No default input device found, trying device 0")
                self.device_index = 0

        self._open_stream()

    def _open_stream(self):
        """Open or re-open the audio stream."""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.close()
            except:
                pass

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
            # FORCE UNMUTE: Ensure mixer is correct every time we open stream
            force_unmute_mic()
            print(f"✅ Audio stream opened successfully")
        except Exception as e:
            print(f"❌ Failed to open audio stream: {e}")
            self.audio.terminate()
            raise

    async def recv(self):
        """Receive audio data."""
        if not self._stream_active:
            return None
        try:
            # Check if stream is still active (may throw if stream closed during disconnect)
            try:
                if not self.stream.is_active():
                    print("⚠️ Audio input stream inactive, attempting restart...")
                    self._open_stream()
            except OSError:
                # Stream was closed (disconnect in progress)
                self._stream_active = False
                return None

            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
            except IOError as e:
                # Handle Input Overflow/Underflow
                print(f"⚠️ Audio read error: {e}")
                data = b'\x00' * (self.chunk * 2) # Return silence

            if len(data) == 0:
                print("⚠️ Received empty audio frame")
                return None

            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Mute if bot is speaking (half-duplex)
            global last_bot_activity_time
            MUTE_DURATION = 1.5 # Keep muted for 1.5s after bot stops

            if (time.time() - last_bot_activity_time) < MUTE_DURATION:
                audio_data = np.zeros_like(audio_data)

            # Reshape for mono channel: AudioFrame expects (channels, samples)
            if len(audio_data) > 0:
                audio_data = audio_data.reshape((1, -1))
            frame = AudioFrame.from_ndarray(
                audio_data,
                format='s16',
                layout='mono'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.timestamp
            self.timestamp += len(audio_data[0]) if len(audio_data) > 0 else self.chunk
            return frame
        except Exception as e:
            print(f"⚠️ Error reading audio: {e}")
            self._stream_active = False
            return None

    def stop(self):
        """Stop the audio stream."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

    def flush(self):
        """Flush the audio input buffer."""
        if hasattr(self, 'stream') and self.stream.is_active():
             try:
                 available = self.stream.get_read_available()
                 if available > 0:
                     self.stream.read(available, exception_on_overflow=False)
             except Exception as e:
                 print(f"⚠️ Error flushing buffer: {e}")


async def create_webrtc_connection(audio_input_device: Optional[str] = None):
    """Create WebRTC connection to the bot server."""
    global peer_connection, pc_id, global_audio_track

    print("🎤 Requesting audio input...")
    
    # Use PyAudio approach (similar to how test.py uses pygame)
    # This works around ALSA configuration issues
    player = None
    
    if PYAUDIO_AVAILABLE:
        try:
            print("🔍 Trying PyAudio for audio input (similar to test.py's pygame approach)...")
            # Create custom audio track using PyAudio
            audio_track = PyAudioTrack()
            global_audio_track = audio_track # Store reference for flushing
            # Create a simple player-like object that provides the audio track
            # We'll add the track directly to the peer connection
            player = type('Player', (), {'audio': audio_track})()
            print("✅ Audio input opened via PyAudio")
        except Exception as e:
            print(f"⚠️ PyAudio failed: {e}")
            player = None
    
    # Fallback: try MediaPlayer with arecord subprocess
    if player is None:
        import subprocess
        CARD_NAME = 'wm8960soundcard'
        
        # Try using arecord (same tool that test.py uses, but for input)
        try:
            print(f"🔍 Trying arecord with hw:{CARD_NAME} (same device as test.py)...")
            # Test if arecord can access the device
            test_cmd = ['arecord', '-D', f'hw:{CARD_NAME}', '-d', '1', '-f', 'S16_LE', '-r', '16000', '-c', '1', '/dev/null']
            result = subprocess.run(test_cmd, capture_output=True, timeout=2)
            if result.returncode == 0:
                # arecord works, use it with MediaPlayer
                # Note: MediaPlayer might not work with stdin, so we'll need a different approach
                # For now, suggest using pyaudio
                print(f"✅ arecord can access hw:{CARD_NAME}, but MediaPlayer needs pyaudio")
                print("💡 Please install pyaudio: pip install pyaudio")
                return False
            else:
                print(f"⚠️ arecord test failed: {result.stderr.decode()}")
        except Exception as e:
            print(f"⚠️ arecord test error: {e}")
    
    if player is None:
        print(f"❌ Failed to open audio input")
        print("\n💡 Solutions:")
        print("   1. Install pyaudio: pip install pyaudio")
        print("   2. Or test arecord manually: arecord -D hw:wm8960soundcard -f S16_LE -r 16000 -c 1 test.wav")
        return False

    # Create audio output (speaker)
    try:
        # Create media player for audio output
        # We'll handle output through the peer connection's audio track
        print("🔊 Setting up audio output...")
    except Exception as e:
        print(f"WARNING: Audio output setup issue: {e}")

    # Create peer connection
    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    )
    
    peer_connection = RTCPeerConnection(configuration=config)
    
    # Track connection state
    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        global lcd_paused
        state = peer_connection.connectionState
        screen_print(f"📡 Connection state: {state}")
        if state == "connected":
            lcd_paused = True  # Pause LCD updates during call
            display_logo()
            board.set_rgb(0, 255, 0)  # Green LED - connected
            force_unmute_mic()

        elif state == "disconnected":
            display_logo()
            board.set_rgb(255, 255, 0)  # Yellow LED

        elif state == "failed":
            lcd_paused = False  # Resume LCD updates
            display_logo()
            board.set_rgb(255, 0, 0)  # Red LED
            refresh_debug_screen()  # Show final state

        elif state == "closed":
            lcd_paused = False  # Resume LCD updates
            display_logo()
            board.set_rgb(128, 128, 128)  # Gray LED
            refresh_debug_screen()  # Show final state

    # Handle ICE candidates
    @peer_connection.on("icecandidate")
    async def on_icecandidate(event):
        if event.candidate and pc_id:
            candidate_dict = {
                "candidate": event.candidate.candidate,
                "sdp_mid": event.candidate.sdpMid,
                "sdp_mline_index": event.candidate.sdpMLineIndex,
            }
            async with aiohttp.ClientSession() as session:
                await send_ice_candidate(candidate_dict, pc_id, session)

    # Handle incoming audio track (bot's voice)
    @peer_connection.on("track")
    def on_track(track):
        print(f"🎵 Received {track.kind} track")
        if track.kind == "audio":
            print("🔊 Bot audio track received - setting up playback...")
            # Start audio playback task
            asyncio.create_task(play_audio_track(track))

    # Create data channel (CLIENT must create it for SmallWebRTC!)
    data_channel = peer_connection.createDataChannel("data", ordered=True)
    
    @data_channel.on("open")
    def on_data_channel_open():
        print("✅ Data channel opened - ready for RTVI messages!")
        display_logo()  # Keep logo visible
        board.set_rgb(0, 0, 255)  # Blue LED - data channel ready

    @data_channel.on("message")
    def on_data_channel_message(message):
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = json.loads(message.decode('utf-8'))
            
            # Handle RTVI messages
            if data.get("label") == "rtvi-ai":
                msg_type = data.get("type")
                text = data.get("data", {}).get("text") or data.get("text")
                
                if msg_type == "user-transcription" and text:
                    screen_print(f"\n👤 User: {text}")
                elif msg_type in ["bot-transcription", "tts-text"] and text:
                    screen_print(f"\n🤖 Bot: {text}")
        except Exception as e:
            print(f"⚠️ Error parsing data channel message: {e}")

    # Add audio transceiver for sending
    if player and hasattr(player, 'audio'):
        # Use the audio track from player
        audio_transceiver = peer_connection.addTransceiver(
            player.audio,
            direction="sendrecv"
        )
        print("✅ Audio transceiver added")
    else:
        print("⚠️ No audio track available")
        return False
    
    # Add video transceiver (required by SmallWebRTC, even if not used)
    peer_connection.addTransceiver("video", direction="sendrecv")

    # Create offer
    print("📤 Creating WebRTC offer...")
    await peer_connection.setLocalDescription(await peer_connection.createOffer())
    
    offer = {
        "sdp": peer_connection.localDescription.sdp,
        "type": peer_connection.localDescription.type
    }

    # Send offer to server
    print(f"📡 Sending offer to {server_url}/api/offer...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{server_url}/api/offer",
            json=offer,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"ERROR: Server error: {response.status} - {error_text}")
                return False

            answer_data = await response.json()
            print(f"✅ Received answer from server")
            
            # Extract pc_id from answer
            pc_id = answer_data.get("pc_id")
            if not pc_id:
                print("ERROR: No pc_id in server response")
                return False
            
            print(f"🔑 Connection ID: {pc_id}")
            
            # Set remote description
            answer = RTCSessionDescription(
                sdp=answer_data["sdp"],
                type=answer_data["type"]
            )
            await peer_connection.setRemoteDescription(answer)
            print("✅ Remote description set")

    return True


async def disconnect_webrtc():
    """Disconnect WebRTC connection."""
    global peer_connection, pc_id, last_bot_activity_time, lcd_paused

    # Resume LCD updates
    lcd_paused = False

    # Reset activity timer so mic opens immediately
    last_bot_activity_time = 0.0

    print("🔌 Disconnecting...")
    if peer_connection:
        # Stop all tracks before closing
        for sender in peer_connection.getSenders():
            if sender.track:
                try:
                    sender.track.stop()
                except:
                    pass
        try:
            await peer_connection.close()
        except:
            pass
        peer_connection = None
    pc_id = None
    display_logo()  # Restore logo
    board.set_rgb(0, 0, 0)  # Turn off LED


async def on_button_pressed(audio_input_device: Optional[str] = None):
    """Handle button press - toggle connection to bot."""
    global peer_connection

    screen_print("\n🔘 Button pressed!")

    # Toggle: If connected, disconnect. If disconnected, connect.
    if peer_connection and peer_connection.connectionState not in ["closed", "failed"]:
        screen_print("🔌 Disconnecting...")
        await disconnect_webrtc()
        screen_print("✅ Disconnected - ready for next conversation")
        return

    # Visual feedback - keep logo visible, just change LED
    display_logo()  # Keep logo visible
    board.set_rgb(255, 255, 0)  # Yellow LED - connecting

    # Connect to bot
    success = await create_webrtc_connection(audio_input_device)
    
    if not success:
        print("❌ Failed to connect to bot")
        display_logo()  # Keep logo visible
        board.set_rgb(255, 0, 0)  # Red LED - error
        await asyncio.sleep(2)
        display_logo()  # Restore logo
        board.set_rgb(0, 0, 0)  # Turn off LED


# Global event loop reference
main_loop: Optional[asyncio.AbstractEventLoop] = None

def setup_button_handler(loop: asyncio.AbstractEventLoop):
    """Setup button press handler."""
    global main_loop, audio_input_device
    main_loop = loop
    
    # Note: WhisPlayBoard.on_button_press expects a synchronous callback
    # We'll schedule the async function on the event loop
    def button_callback():
        if main_loop and main_loop.is_running():
            # Schedule the coroutine on the running event loop
            asyncio.run_coroutine_threadsafe(on_button_pressed(audio_input_device), main_loop)
        else:
            print("⚠️ Event loop not available")
    
    board.on_button_press(button_callback)
    print("✅ Button handler registered")


def disable_hardware_loopback():
    """Disable WM8960 hardware loopback paths using amixer."""
    # This function is currently disabled as it may mute the mic
    pass
    # import subprocess
    # controls = [
    #     ('Left Output Mixer PCM', 'on'),  # Ensure PCM is routed to output
    #     ('Right Output Mixer PCM', 'on'),
    #     ('3D', '0'),          # Disable 3D enhancement (can cause bleed)
    #     ('ALC', 'off'),       # Disable Automatic Level Control
    #     ('ADC PCM', '0'),     # Disable Loopback if present (check specific control name)
    # ]
    # 
    # print("🔇 Configuring hardware mixer to prevent loopback...")
    # for control, value in controls:
    #     try:
    #         subprocess.run(
    #             ['amixer', '-D', 'hw:wm8960soundcard', 'sset', control, value],
    #             capture_output=True, check=False
    #         )
    #     except:
    #         pass

def force_unmute_mic():
    """Ensure microphone is unmuted in hardware mixer."""
    import subprocess
    print("🔊 Forcing microphone unmute...")
    controls = [
        ('Capture', 'cap'),      # Enable capture
        ('Capture', '63'),       # Max gain
        ('ADC PCM', '255'),      # Max ADC digital volume (0 = mute, 255 = max)
        ('L Input Mux', 'L2'),   # Select Mic input (L2/R2 typically)
        ('R Input Mux', 'R2'),
        ('Left Input Boost Mixer LINPUT2', 'on'), # Boost
        ('Right Input Boost Mixer RINPUT2', 'on'),
        ('3D', '0'),             # Disable 3D enhancement
        ('ALC', 'off')           # Disable ALC
    ]
    
    for control, value in controls:
        try:
            subprocess.run(
                ['amixer', '-D', 'hw:wm8960soundcard', 'sset', control, value],
                capture_output=True, check=False
            )
        except:
            pass

async def main_loop():
    """Main event loop."""
    global logo_image_data
    
    screen_print("🤖 Raspberry Pi Bot Ready")
    
    # Ensure mic is unmuted
    force_unmute_mic()
    
    # Try to disable loopback
    # disable_hardware_loopback()  <-- Commented out to prevent accidental mic mute
    
    screen_print(f"📡 Server URL: {server_url}")
    if audio_input_device:
        screen_print(f"🎤 Audio input device: {audio_input_device}")
    else:
        screen_print("🎤 Audio input device: Auto-detect")
    screen_print("⏳ Waiting for button press...")
    screen_print("")
    
    # Load and display logo
    logo_path = os.path.join(os.path.dirname(__file__), "barelabs_ai_logo.png")
    if os.path.exists(logo_path):
        try:
            logo_image_data = load_jpg_as_rgb565(logo_path, board.LCD_WIDTH, board.LCD_HEIGHT)
            display_logo()
            print(f"✅ Logo displayed: {os.path.basename(logo_path)}")
        except Exception as e:
            print(f"⚠️ Failed to load logo: {e}")
    else:
        print(f"⚠️ Logo not found: {logo_path}")
    
    # Set initial volume
    set_wm8960_volume_stable("121")
    
    # Setup button handler with current event loop
    loop = asyncio.get_event_loop()
    setup_button_handler(loop)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        await disconnect_webrtc()
        board.cleanup()
        print("👋 Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Bot - Connect to WebRTC server on button press"
    )
    parser.add_argument(
        "--server",
        default="http://192.168.1.213:7860",
        help="Server URL (default: http://192.168.1.213:7860)"
    )
    parser.add_argument(
        "--audio-input",
        default=None,
        help="Audio input device (e.g., hw:0,0, plughw:wm8960soundcard). If not specified, will auto-detect."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--debug-screen",
        action="store_true",
        help="Show debug printouts on LCD screen instead of logo"
    )
    args = parser.parse_args()

    # If list devices, show them and exit
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Set global variables from args
    server_url = args.server
    audio_input_device = args.audio_input
    if args.debug_screen:
        SHOW_DEBUG_ON_SCREEN = True
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        board.cleanup()
        sys.exit(0)

