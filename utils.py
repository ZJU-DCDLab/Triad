import logging
import os.path
from datetime import datetime
import time
from collections import defaultdict


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


time_dict = defaultdict(list)

def print_time_dict(td):
    temp_dict = {k: round(sum(v), 5) for k, v in td.items()}
    print(temp_dict)
    logger.info(f"{temp_dict}")

def log_time_cost(method_name, reset=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            time_dict[method_name].append(execution_time)
            log = "{} cost: {:.2f} s".format(method_name, execution_time)
            logger.info(log)

            if reset:
                print_time_dict(time_dict)
                time_dict.clear()
            return result
        return wrapper
    return decorator
