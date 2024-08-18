import openai
from configparser import ConfigParser
from utils import log_lm, log_time_cost
import time
import transformers
import importlib

# global api using times
openai_api_use_times = 0
openai_api_success_times = 0
openai_api_fail_times = 0

openai_input_tokens = 0
openai_output_tokens = 0


class OpenAI_Language_Model(object):
    def __init__(self, config: ConfigParser) -> None:
        """language model using OpenAI's API

        Args:
            base_model (str, optional): base models from OpenAI. Defaults to 'gpt-3.5-turbo'.
            generator_type (str, optional): using chat-completion or text-completion. Defaults to 'chat-completion'.
        """
        self.base_model = config.get('openai', 'base_model')
        self.generator_type = config.get('openai', 'generator_type')
        self.temperature = float(config.get('openai', 'temperature'))
        openai.api_key_path = config.get('openai', 'api_key_path') 
        openai.api_base = config.get('openai', 'api_base') 
        openai.proxy = config.get('openai', 'proxy') 
    
    @log_lm
    @log_time_cost("LM Generation")
    def generate(self, prompt:str, decode_fn=lambda x:x) -> str:
        """generate texts based on given prompt

        Args:
            prompt (str): prompt for generation
            decode_fn (function, optional): decoding function, helpful when CoT enabled . Defaults to lambda x:x.

        Returns:
            str: _description_
        """
        global openai_api_use_times, openai_api_success_times, openai_api_fail_times, openai_input_tokens, openai_output_tokens
        while True: # retry while openai is busy or something wrong 
            try:
                if self.generator_type == 'text-completion':
                    openai_api_use_times += 1
                    res = openai.Completion.create(
                        model=self.base_model,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=2048,
                        timeout=120
                    )
                    query_result = res['choices'][0]['text']
                elif self.generator_type == 'chat-completion':
                    openai_api_use_times += 1
                    result = openai.ChatCompletion.create(
						model=self.base_model,
						messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        timeout=120)
                    query_result = result['choices'][0]['message']['content']
                # if cot format
                res = decode_fn(query_result)
                openai_api_success_times += 1
                openai_input_tokens += result["usage"]["prompt_tokens"]
                openai_output_tokens += result["usage"]["completion_tokens"]
                return res
            except openai.error.InvalidRequestError as e:
                print(e)
                openai_api_fail_times += 1
                raise ValueError(str(e))
            except openai.error.RateLimitError as e:
                print(e)
                exit(0)
            except openai.OpenAIError as e:
                print(e)
                time.sleep(15)
                openai_api_fail_times += 1

class HuggingFaceModel():
    def __init__(self, config: ConfigParser) -> None:
        model = getattr(transformers, config.get('huggingface', 'class'))
        self.model = model.from_pretrained(
            config.get('huggingface', 'model_path'),
            device_map=config.get('huggingface', 'device'),
            trust_remote_code=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.get('huggingface', 'model_path'),
            trust_remote_code=True
        )
        self.replace_brackets = config.get('huggingface', 'class') == 'T5ForConditionalGeneration'
    
    @log_lm
    def generate(self, prompt:str, decode_fn=lambda x:x) -> str:
        if self.replace_brackets:
            prompt = prompt.replace('<', '[').replace('>', ']')
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(inputs.input_ids, max_length=4096, do_sample=True, temperature=0.2)
        query_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.replace_brackets:
            query_result = query_result.replace('[', '<').replace(']', '>')
        else:
            query_result = query_result.strip(prompt)
        res = decode_fn(query_result)
        return res

if __name__ == '__main__':
    from config import ConfigParser
    config = ConfigParser()
    config.read('conf/lc-quad-1.0.conf')
    lm = config.get_lm(model='huggingface')
    prompt = "Hello!"
    print(lm.generate(prompt))
