import paramiko
import sys
import os

def upload_file(local_path, remote_path):
    ip = "192.168.137.151"
    username = "admin"
    password = "admin"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip, username=username, password=password, timeout=10)
        sftp = ssh.open_sftp()
        
        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            try:
                sftp.stat(remote_dir)
            except IOError:
                # Directory doesn't exist, try to make it
                sftp.mkdir(remote_dir)
                
        print(f"Uploading {local_path} to {remote_path}...")
        sftp.put(local_path, remote_path)
        print("[SUCCESS] Upload complete!")
        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upload_remote.py <local_path> <remote_path>")
        sys.exit(1)
    upload_file(sys.argv[1], sys.argv[2])
