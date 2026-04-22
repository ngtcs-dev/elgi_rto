# elgi_rto

# Initial setup
python3 -m venv .venv310
source .venv310/bin/activate
pip install -r requirements.txt

# Run
python main.py

# Build
pyinstaller main.spec

# Notes
- Works on both macOS and Windows.
- On Windows, activate with `.venv310\\Scripts\\activate`.
- `config.ini` and relative capture paths are resolved from the application folder.
- Set `forced_state_code = OD` in `config.ini` when all registration numbers are expected to start with `OD`.
