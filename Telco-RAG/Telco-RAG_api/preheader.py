# Import this module into any TelcoRAG.py file that u will be eventually running
import sys
import os
curr_path = os.path.abspath(__file__)
VENV_PATH = os.environ.get('VIRTUAL_ENV') # virtual environment path

ROOT_DIR = "Telco-RAG_api"
ROOT_PATH = curr_path[:curr_path.find(ROOT_DIR) + len(ROOT_DIR)]
sys.path.append(ROOT_PATH)

# These are env var for this project under the root dir (modify accordingly)
CLONE_DIR = "3GPP-Release18"
CLONE_PATH = os.path.join(ROOT_PATH, CLONE_DIR)
# Keeping Download directory outside git-repo to avoid of loss of data when local repo is refreshed
DOWN_DIR = "3GPP-Latest"
DOWN_PATH = os.path.join(VENV_PATH, DOWN_DIR) # could keep inside Root_dir as well

# Special logging functionalities: colorful log in terminal + backup log into file
import logging
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"    # DEBUG
    cyan = "\x1b[36;15m"    # INFO
    yellow = "\x1b[33;10m"  # WARNING
    magenta = "\x1b[35;5m"  # ERROR
    red = "\x1b[31;1m"      # FATAL
    white = "\x1b[37;20m"   # MESSAGE
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    #note: logger name to be set based on filename for easy to locate
    fmt_lvl = "[%(levelname).3s | %(name)s:%(lineno)s - "
    fmt_fn = "%(funcName)s()] " 
    msg = "%(message)s"
    _format = fmt_lvl + fmt_fn + msg

    FORMATS = {
        logging.DEBUG: grey + fmt_lvl + green + fmt_fn + reset + msg,
        logging.INFO: cyan + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.WARNING: yellow + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.ERROR: magenta + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.CRITICAL: red + fmt_lvl + green + fmt_fn + white + msg + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# globaL format refernce
cf = CustomFormatter()
FORMAT = cf._format

# Console handler - with a higher log level
CONSOLE = logging.StreamHandler()
CONSOLE.setLevel(logging.DEBUG)
CONSOLE.setFormatter(cf)

# File handler - overwrites log file every time
LOG_FILE = "TelcoRAG.log"
FILE_HANDLER = logging.FileHandler(LOG_FILE, mode='w')  # 'w' means overwrite each run
FILE_HANDLER.setLevel(logging.DEBUG)
FILE_HANDLER.setFormatter(logging.Formatter(FORMAT))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
# Log into terminal as well as File, so to entire log, (gets trucated in terminal in some cases)
root_logger.addHandler(CONSOLE)
root_logger.addHandler(FILE_HANDLER)
