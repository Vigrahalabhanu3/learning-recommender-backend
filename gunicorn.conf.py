import os

# Render injects PORT env variable — bind to it
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")
workers = 1
timeout = 120
