'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
evaluator.py

python -m evaluator \
    --config_path 'lc-quad-1.0.conf' \
    --samples 200
'''
from question_parser import QuestionParser
from query_generator import QueryGenerator
from answer_generator import AnswerGenerator
from datetime import datetime
from tqdm import tqdm
from config import ConfigParser
from utils import enable_global_logging_config, time_dict
import os
import json
import fire
import lm


class Evaluator():
    def __init__(self, 
                 config_path: str,
                 samples: int=None
                ) -> None:
        config = ConfigParser()
        if not os.path.exists(os.path.join('conf', config_path)):
            raise FileNotFoundError('Config not exists: ' + os.path.join('conf', config_path))
        config.read(os.path.join('conf', config_path))
        self.config = config

        self.samples = samples
        self.dataset = config.get('evaluate', 'dataset')
        self.test_cases = dict()
        self.retry_times = int(config.get('evaluate', 'retry_times'))

        self.output_file = os.path.join('results', config_path + f'-{samples}-{datetime.now()}.json')
        enable_global_logging_config(log_file_name=config_path + f'-{samples}-{datetime.now()}.log')
        
        if self.dataset == 'LC-QuAD-1.0':
            self._load_LC_QuAD_1_0()
        elif self.dataset == 'QALD-9':
            self._load_QALD_9()
        elif self.dataset == 'DBLP':
            self._load_DBLP()
        elif self.dataset == 'YAGO':
            self._load_YAGO()
        
        self.question_parser = QuestionParser(config)
        self.query_generator = QueryGenerator(config)
        self.answer_generator = AnswerGenerator(config)

    def _load_LC_QuAD_1_0(self):   
        with open(os.path.join('benchmarks', 'LC-QuAD-1.0', 'test-data.json'), 'r') as f:
            data = json.load(f)
            for example in data:
                self.test_cases[example['_id']] = {
                    'question': example['corrected_question'],
                    'sparql': example['sparql_query']
                }
                if len(self.test_cases) == self.samples:
                    break 

    def _load_QALD_9(self):        
        with open(os.path.join('benchmarks', 'QALD-9', 'qald-9-test-multilingual.json'), 'r') as f:
            data = json.load(f)
            for example in data['questions']:
                for instance in example['question']:
                    if instance['language'] == 'en':
                        self.test_cases[example['id']] = {
                            'question': instance['string'],
                            'sparql': example['query']['sparql']
                        }
                        break
                if len(self.test_cases) == self.samples:
                    break
    
    def _load_DBLP(self):        
        with open(os.path.join('benchmarks', 'DBLP', 'qald9_dblp100.json'), 'r') as f:
            data = json.load(f)
            for example in data['questions']:
                for instance in example['question']:
                    if instance['language'] == 'en':
                        self.test_cases[example['id']] = {
                            'question': instance['string'],
                            'sparql': example['query']['sparql']
                        }
                        break
                if len(self.test_cases) == self.samples:
                    break
    
    def _load_YAGO(self):        
        with open(os.path.join('benchmarks', 'yago', 'qald9_yago100.json'), 'r') as f:
            data = json.load(f)
            for example in data['questions']:
                for instance in example['question']:
                    if instance['language'] == 'en':
                        self.test_cases[example['id']] = {
                            'question': instance['string'],
                            'sparql': example['query']['sparql']
                        }
                        break
                if len(self.test_cases) == self.samples:
                    break
                
    def save(self, results):
        config = {}
        for section, _ in self.config.items():
            temp = {}
            for key, val in self.config.items(section):
                temp[key] = val
            config[section] = temp
        results['config'] = config
        results['samples'] = self.samples
        results['openai_usage'] = {
            "openai_api_use_times": lm.openai_api_use_times,
            "openai_api_success_times": lm.openai_api_success_times,
            "openai_api_fail_times": lm.openai_api_fail_times,
            "openai_input_tokens": lm.openai_input_tokens,
            "openai_output_tokens": lm.openai_output_tokens
        }
        results['test_cases'] = self.test_cases

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def evaluate(self):
        precions = []
        recalls = []
        f1s = []
        acc = 0

        for index, (idx, val) in tqdm(enumerate(self.test_cases.items()), total=len(self.test_cases)):
            question, gold_sparql = val['question'], val['sparql']
            result = self.evaluate_sample(idx, question, gold_sparql)
            precions.append(result['precision'])
            recalls.append(result['recall'])
            f1s.append(result['f1'])
            acc += result['accuracy']
            self.save(dict())

        results = {
            'test_number': len(self.test_cases),
            'average precision': round(sum(precions) / len(precions), 4),
            'average recall': round(sum(recalls) / len(recalls), 4),
            'average f1': round(sum(f1s) / len(f1s), 4),
            'accuracy': round(acc / len(self.test_cases), 4)
        }
        self.save(results)
        return results

    def recall_score(self, true_entities, predict_entities):
        TP = len(set(true_entities) & set(predict_entities))
        return round(TP / (len(true_entities) + 1e-5), 4)

    def precision_score(self, true_entities, predict_entities):
        TP = len(set(true_entities) & set(predict_entities))
        return round(TP / (len(predict_entities) + 1e-5), 4)

    def f1_score(self, true_entities, predict_entities):
        r = self.recall_score(true_entities, predict_entities)
        p = self.precision_score(true_entities, predict_entities)
        return round(p*r*2 / (p+r + 1e-5), 3)
    
    def accuracy(self, true_entities, predict_entities):
        if set(true_entities) == set(predict_entities):
            return 1.0
        return 0.0

    def evaluate_sample(self, idx, question, gold_sparql):
        gold_entities = self.answer_generator.get_result(gold_sparql) 
        generated_entities = []

        for i in range(self.retry_times):
            self.test_cases[idx]['tries'] = self.test_cases[idx].get('tries', 0) + 1
            try:
                if self.answer_generator.answer_generator_version == 'v0':
                    generated_entities = self.answer_generator.answer_generate(question=question)
                    self.test_cases[idx]['generated_entities'] = generated_entities
                else:
                    parse_result = self.question_parser.parse(question)
                    generated_sqarql, prompt = self.query_generator.generate(input=parse_result, return_prompt=True)
                    if generated_sqarql == '<no sparql>' and i < (self.retry_times-1):
                        continue
                    generated_entities = self.answer_generator.answer_generate(
                        sparql = generated_sqarql,
                        question_type = parse_result.task_type,
                        question = parse_result.question
                    )
                    if len(generated_entities) == 0:
                        continue
                    if 'exception' in self.test_cases[idx]:
                        self.test_cases[idx].pop('exception')
                    self.test_cases[idx]['question_type'] = parse_result.task_type
                    self.test_cases[idx]['triples'] = parse_result.triples
                    self.test_cases[idx]['prompt'] = prompt
                    self.test_cases[idx]['generated_sqarql'] = generated_sqarql
                    break
            except Exception as e:
                print(e)
                self.test_cases[idx]['exception'] = str(e)
        precision = self.precision_score(gold_entities, generated_entities)
        recall = self.recall_score(gold_entities, generated_entities)
        f1 = self.f1_score(gold_entities, generated_entities)
        accuracy = self.accuracy(gold_entities, generated_entities)

        self.test_cases[idx]['precision'] = precision 
        self.test_cases[idx]['recall'] = recall 
        self.test_cases[idx]['f1'] = f1 
        self.test_cases[idx]['accuracy'] = accuracy
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }


def main(**kwargs):
    evaluator = Evaluator(**kwargs)
    evaluator.evaluate()

if __name__ == "__main__":
    fire.Fire(main)
