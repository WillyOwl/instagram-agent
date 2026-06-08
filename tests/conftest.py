"""tests/conftest.py — Set up test environment.

Sets up mock environment variables so that config.py can be imported
without raising ValueError.
"""

import os

# Set dummy env vars for all tests
os.environ["INSTAGRAM_USERNAME"] = "dummy_user"
os.environ["INSTAGRAM_PASSWORD"] = "dummy_pass"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "llama3"
os.environ["SENDER_WHITELIST"] = "alice,bob"
os.environ["HISTORY_PATH"] = "data/instagram_export/messages/"
os.environ["TESTING"] = "true"

