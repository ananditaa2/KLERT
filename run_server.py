import os
import sys
import subprocess

def main():
    print("==================================================")
    print("  KLERT Brain Scan Analysis Project")
    print("  Starting Local Web Server on http://127.0.0.1:8000")
    print("==================================================")
    
    # Check for virtual environment python
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_bin = venv_python
    else:
        python_bin = sys.executable

    try:
        subprocess.run([python_bin, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n[!] Server stopped cleanly.")
    except Exception as e:
        print(f"\n[!] Error starting server: {e}")

if __name__ == "__main__":
    main()
