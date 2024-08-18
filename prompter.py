'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
prompter.py
'''
import os
import re
import json
from typing import Any, List


class Prompter:
    def __init__(self, task: str, shots:int =0, cot=False) -> None:
        assert isinstance(shots, int)
        assert shots >= 0
        assert task in ['ner', 'rel_cls', 'triple_iden', 'var_iden', \
                        'ans_cls', 'sparql_v1', 'sparql_v2', 'sparql_v3', 'sparql_v4', 'predicate_linking', 'sparql_choose', \
                        'entity_linking_v1', 'entity_linking_v2','entity_linking_v3','entity_linking_v4', 'entity_linking_v5', \
                        'yesorno', 'answer_generate_v3', 'answer_generate_v0'], f"Do not support {task}"
        self.task = task
        self.shots = shots
        self.cot = cot

        # read instructions templates
        with open(os.path.join('prompts', f'{self.task}.txt'), 'r') as f:
            self.prompt = f.read()
            self.prompt += '\n'

        # read few-shots data and CoT data
        with open(os.path.join('few-shots', f'{self.task}.json'), 'r') as f:
            data = json.load(f)
        
        # if CoT enabled, since the output will be a chain of thought, we will need \
        # to extract target information using some templates
        if cot:
            data = data['CoT']
            self.cot_target = data['CoT-target']
        # CoT not enabled
        else:
            data = data['normal']
        
        # target -- output template
        # few-shots -- few-shots data 
        target, few_shots = data['target'], data['few-shots']

        # if few-shots enabled, append few-shot prompts to the end 
        if self.shots > 0:
            self.prompt += 'Here are some examples: \n'
            # append examples to the end, splitting by '###'
            for shot in few_shots[:self.shots]: # enable few shots
                self.prompt += '###\n'
                for key, value in shot.items():
                    if isinstance(value, list):
                        value = '\n' + '\n'.join(value)
                    self.prompt += f'{key}: {value} \n'
            self.prompt += '###\n\n'
            self.prompt += "Now, it is your turn:\n"
        
        # append output template 
        for key, value in target.items():
            self.prompt += f'{key}: {value} \n'
        
    def __call__(self, *args: Any, **kwds: Any) -> str:
        """the method to format the prompt template using given parameters

        Returns:
            str: full prompt after format
        """
        return self.prompt.format(*args)
    
    def decode_output(self, output: str) -> str:
        """decode output since there are something useless when cot enabled

        Args:
            output (str): output before decoding

        Returns:
            str: output after decoding
        """
        # if CoT enabled
        if self.cot:
            output = re.search(self.cot_target, output).group(1)
        # otherwise
        return output
    
if __name__ == '__main__':
    prompter = Prompter(task='triple_iden', shots=3, cot=True)
    print(prompter("How many movies did Stanley Kubrick direct?"))
