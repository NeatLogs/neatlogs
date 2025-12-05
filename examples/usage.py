import os
import logging
from neatlogs import init

# Point to a local server (or use a mock sink)
os.environ["NEATLOGS_API_URL"] = "http://localhost:3000"

# Initialize
init(api_key="test-key")

logging.info("This is a test log from the local playground")
print("Trace sent (check your local server console)")
