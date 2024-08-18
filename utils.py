import logging
import os.path
from datetime import datetime


def enable_global_logging_config(log_dir: str = './log', log_file_name=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_file_name is not None:
        log_file_name = os.path.join(log_dir, log_file_name)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logfile = log_file_name or os.path.join(log_dir, date + '.log') 
    formatter = "%(asctime)s - [%(filename)s:line:%(lineno)d] - %(levelname)s >> %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO,
                        handlers=[logging.FileHandler(filename=logfile, mode='a')])

logger = logging.getLogger()

def format_lm_input_and_response(input, response):
    return f"\nPrompt for model: >>>>>>>>\n{input}\n<<<<<<<<\n\nResponse of model: >>>>>>>>\n{response}\n<<<<<<<<\n"

def log_lm(func):
    def wrapper(*args, **kwargs):
        prompt = args[1] or kwargs['prompt']
        response = func(*args, **kwargs)
        logger.info(format_lm_input_and_response(prompt, response))
        return response
    return wrapper
