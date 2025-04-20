import logging
import sys
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config() -> dict:
    """Loads the configuration from config.yaml"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError("Config file is empty or invalid.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {CONFIG_PATH}")
        sys.exit(1) # Exit if config is crucial and missing
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)


def setup_logging(config: dict):
    """Sets up basic logging based on configuration."""
    log_level_str = config.get('logging', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout, # Log to console
        # filename='app.log', # Uncomment to log to a file
        # filemode='a'       # Append mode for file logging
    )

    # Optionally silence verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Level: {log_level_str}")
    return logger # Return a logger instance for the caller

# Load config once at module level for potential reuse
try:
    CONFIG = load_config()
except SystemExit:
    # Handle exit gracefully if config loading fails during import
    CONFIG = {} # Provide empty dict to avoid further import errors
