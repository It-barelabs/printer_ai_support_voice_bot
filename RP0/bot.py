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
    """Display the logo on the screen and keep it visible."""
    global logo_image_data
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
    if not PYAUDIO_AVAILABLE:
        return
    
    # Using 16000Hz as requested
    TARGET_RATE = 16000 
    CARD_NAME = 'wm8960soundcard'
    audio = pyaudio.PyAudio()
    
    # Find device index
    device_index = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if CARD_NAME.lower() in info['name'].lower() and info['maxOutputChannels'] > 0:
            device_index = i
            break

    # 1. Open stream at 16kHz STEREO
    # The WM8960 almost always requires 2 channels at the driver level
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=2, # Still use 2 channels
            rate=TARGET_RATE,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=320 # 20ms buffer for 16kHz
        )
    except Exception as e:
        print(f"❌ Could not open 16kHz stream: {e}")
        return

    try:
        while True:
            frame = await track.recv()
            if frame is None: break
            
            # 2. Get Mono data (16-bit)
            array = frame.to_ndarray()
            mono_data = array[0] if len(array.shape) > 1 else array
            ## breakpoint() 
            if mono_data.dtype == np.float32:
                mono_data = (mono_data * 32767).astype(np.int16)
            
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
            stream.write(mono_data.tobytes(), exception_on_underflow=False)

    finally:
        if stream: stream.close()
        audio.terminate()
 




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
        self.chunk = 320  # 20ms chunks at 16kHz
        self.timestamp = 0  # Track presentation timestamp
        self._stream_active = True  # Track if stream is active
        
        # Try to find the correct input device
        device_index = None
        
        # First, try to find wm8960soundcard by name
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if CARD_NAME.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
                    device_index = i
                    print(f"✅ Found audio input device: {info['name']} (index {i})")
                    break
            except:
                continue
        
        # If not found, use default input device
        if device_index is None:
            try:
                device_index = self.audio.get_default_input_device_info()['index']
                print(f"✅ Using default audio input device (index {device_index})")
            except:
                print("⚠️ No default input device found, trying device 0")
                device_index = 0
        
        # Open audio stream
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk
            )
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
            # Check if stream is still active
            if not self.stream.is_active():
                print("⚠️ Audio input stream is not active")
                self._stream_active = False
                return None
            
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            if len(data) == 0:
                return None
                
            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Reshape for mono channel: AudioFrame expects (channels, samples)
            # For mono: shape should be (1, samples) not (samples, 1)
            if len(audio_data) > 0:
                audio_data = audio_data.reshape((1, -1))  # (channels=1, samples)
            frame = AudioFrame.from_ndarray(
                audio_data,
                format='s16',
                layout='mono'
            )
            frame.sample_rate = self.sample_rate
            # Set presentation timestamp (PTS) based on sample rate
            # PTS is in time_base units, typically 1/sample_rate
            frame.pts = self.timestamp
            # Increment timestamp by number of samples in this frame
            self.timestamp += len(audio_data[0]) if len(audio_data) > 0 else self.chunk
            return frame
        except Exception as e:
            print(f"⚠️ Error reading audio: {e}")
            import traceback
            traceback.print_exc()
            self._stream_active = False
            return None

    def stop(self):
        """Stop the audio stream."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()


async def create_webrtc_connection(audio_input_device: Optional[str] = None):
    """Create WebRTC connection to the bot server."""
    global peer_connection, pc_id

    print("🎤 Requesting audio input...")
    
    # Use PyAudio approach (similar to how test.py uses pygame)
    # This works around ALSA configuration issues
    player = None
    
    if PYAUDIO_AVAILABLE:
        try:
            print("🔍 Trying PyAudio for audio input (similar to test.py's pygame approach)...")
            # Create custom audio track using PyAudio
            audio_track = PyAudioTrack()
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
        state = peer_connection.connectionState
        print(f"📡 Connection state: {state}")
        if state == "connected":
            display_logo()  # Keep logo visible
            board.set_rgb(0, 255, 0)  # Green LED - connected
        elif state == "disconnected":
            # Don't auto-close - connection might reconnect or stay open for next interaction
            display_logo()  # Keep logo visible
            board.set_rgb(255, 255, 0)  # Yellow LED - disconnected but keeping connection
            print("⚠️ Connection disconnected, but keeping peer connection open for potential reconnect")
        elif state == "failed":
            # Only close on actual failure, not on temporary disconnects
            display_logo()  # Keep logo visible
            board.set_rgb(255, 0, 0)  # Red LED - failed
            print("❌ Connection failed")
            # Don't auto-close - let user decide to reconnect via button
        elif state == "closed":
            # Connection was closed (may be by server after bot response)
            # This is normal - connection can be reopened for next interaction
            display_logo()  # Keep logo visible
            board.set_rgb(128, 128, 128)  # Gray LED - connection closed, ready for next
            print("🔌 Connection closed (ready for next interaction)")
            # Don't clear peer_connection - it's already closed, user can reconnect via button

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
                    print(f"👤 User: {text}")
                elif msg_type in ["bot-transcription", "tts-text"] and text:
                    print(f"🤖 Bot: {text}")
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
    global peer_connection, pc_id
    
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
    """Handle button press - connect to bot."""
    global peer_connection
    
    print("\n🔘 Button pressed!")
    
    # Visual feedback - keep logo visible, just change LED
    display_logo()  # Keep logo visible
    board.set_rgb(255, 255, 0)  # Yellow LED - connecting
    
    # If already connected, disconnect first
    if peer_connection and peer_connection.connectionState not in ["closed", "failed"]:
        print("⚠️ Already connected, disconnecting first...")
        await disconnect_webrtc()
        await asyncio.sleep(0.5)
    
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


async def main_loop():
    """Main event loop."""
    global logo_image_data
    
    print("🤖 Raspberry Pi Bot Ready")
    print(f"📡 Server URL: {server_url}")
    if audio_input_device:
        print(f"🎤 Audio input device: {audio_input_device}")
    else:
        print("🎤 Audio input device: Auto-detect (will try multiple devices)")
    print("⏳ Waiting for button press (Press Ctrl+C to exit)...")
    print("")
    
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
    args = parser.parse_args()
    
    # If list devices, show them and exit
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
    
    server_url = args.server
    audio_input_device = args.audio_input
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        board.cleanup()
        sys.exit(0)

