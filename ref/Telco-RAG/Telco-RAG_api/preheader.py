# Import this module into any TelcoRAG.py file that u will be eventually running
import sys
import os
curr_path = os.path.abspath(__file__)
VENV_PATH = os.environ.get('VIRTUAL_ENV') # virtual environment path

REPO_DIR = "Telco-RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]
ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, ROOT_DIR)
sys.path.append(ROOT_PATH)

# These are env var for this project under the root dir (modify accordingly)
RELEASE_VER = 18
CLONE_DIR = f"3GPP-Release{RELEASE_VER}"
CLONE_PATH = os.path.join(ROOT_PATH, CLONE_DIR)
# Keeping Download directory outside git-repo to avoid of loss of data when local repo is refreshed
DOWN_DIR = "3GPP-Latest"
DOWN_PATH = os.path.join(VENV_PATH, DOWN_DIR) # could keep inside Root_dir as well

# Update abs path to 'TeleQnA.txt' dataset here that can be used to random test the rag model
TeleQA_PATH = os.path.normpath(os.path.join(REPO_PATH, "../TeleQnA/TeleQnA.txt"))

# set rand.seed value manually for reproducibility
import random
import datetime
SEED = int(datetime.datetime.now().timestamp())
random.seed(SEED)

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
    fmt_lvl = "[%(levelname).3s | %(name)s:%(lineno)s"
    fmt_fn = " - %(funcName)s()] " 
    msg = "%(message)s"
    _format = fmt_lvl + fmt_fn + msg
    _simple = fmt_lvl+ "] " + msg

    FORMATS = {
        logging.DEBUG: grey + msg + reset, # debug will print only msg with no formats
        logging.INFO: cyan + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.WARNING: yellow + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.ERROR: magenta + fmt_lvl + green + fmt_fn + white + msg + reset,
        logging.CRITICAL: red + fmt_lvl + green + fmt_fn + white + msg + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class FileFormatter(logging.Formatter):
    def __init__(self, full_fmt, debug_fmt):
        self.full_fmt = full_fmt
        self.debug_fmt = debug_fmt
        super().__init__(full_fmt)  # Initialize with full format

    def format(self, record):
        # Switch the format based on log level
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_fmt  # Update the style's format
        else:
            self._style._fmt = self.full_fmt   # Revert to full format
        return super().format(record)

# globaL format refernce
cf = CustomFormatter()
FORMAT = cf._format

# Console handler - with a higher log level
CONSOLE = logging.StreamHandler()
CONSOLE.setLevel(logging.DEBUG)
CONSOLE.setFormatter(cf)

# File handler - overwrites log file every time
LOG_FILE = os.path.dirname(os.getcwd()) + ".log"
FILE_HANDLER = logging.FileHandler(LOG_FILE, mode='w')  # 'w' means overwrite each run
FILE_HANDLER.setLevel(logging.DEBUG)
FILE_HANDLER.setFormatter(FileFormatter(full_fmt=cf._format, debug_fmt=cf.msg))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
# Log into terminal as well as File, so to entire log, (gets trucated in terminal in some cases)
root_logger.addHandler(CONSOLE)
root_logger.addHandler(FILE_HANDLER)
