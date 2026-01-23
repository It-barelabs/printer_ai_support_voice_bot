import os
import sys

try:
    import paramiko
except ImportError:
    paramiko = None

# --- Configuration ---
HOST = "192.168.1.206"       # Replace with your remote IP
USERNAME = "itay"              # Replace with your remote username
PASSWORD = "mk47Ba11" # Remote password
REMOTE_PATH = "/home/itay/Whisplay/example/bot_half_duplex.py" # Remote destination path
LOCAL_FILE = "bot_half_duplex.py"        # Local file to upload
# ---------------------

def sync_to_remote():
    # Check if local file exists
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: Local file '{LOCAL_FILE}' not found.")
        return

    print(f"Preparing to upload '{LOCAL_FILE}' to {USERNAME}@{HOST}:{REMOTE_PATH}")

    try:
        # Create Transport
        print("Connecting...")
        transport = paramiko.Transport((HOST, 22))
        transport.connect(username=USERNAME, password=PASSWORD)
        
        # Create SFTP client
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # Upload file
        print(f"Uploading {LOCAL_FILE}...")
        sftp.put(LOCAL_FILE, REMOTE_PATH)
        
        # Get file stats to confirm
        remote_stat = sftp.stat(REMOTE_PATH)
        print(f"Success! File uploaded. Size: {remote_stat.st_size} bytes.")
        
        sftp.close()
        transport.close()
        
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your password.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if paramiko is None:
        print("Error: 'paramiko' library is required.")
        print("Please install it running: pip install paramiko")
        sys.exit(1)

    sync_to_remote()

