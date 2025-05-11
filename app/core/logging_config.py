import logging
import sys
from app.core.config import settings

def setup_logging():
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # You can add file handlers or more sophisticated logging here
    # e.g., logging.getLogger("uvicorn.access").disabled = True to reduce noise