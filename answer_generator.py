from config import ConfigParser
from lm import OpenAI_Language_Model
from typing import List
from prompter import Prompter
from kb_linker import KB_Linker
from query_executor import QueryExecutor
from utils import log_time_cost


class AnswerGenerator(object):
    def __init__(self, config: ConfigParser) -> None:
        self.answer_generator_version = config.get('answer-generate', 'answer_generate_version')
        self.yesorno_prompter = Prompter(task='yesorno')
        if self.answer_generator_version in ['v0', 'v3']:
            self.prompter = Prompter(task=f'answer_generate_{self.answer_generator_version}')
        self.lm = config.get_lm(config.get('answer-generate', 'model'))
        self.linker = KB_Linker(config)
        self.query_executor = QueryExecutor(config)
    
    def get_result(self, sparql):
        if self.query_executor.is_sparql_correct(sparql):
            generated_entities = self.query_executor.execute(sparql)
            generated_entities = self._extract_entities(generated_entities)
        else:
            generated_entities = []
        return generated_entities

    def _extract_entities(self, result):
        if isinstance(result, list):
            results = []
            for sample in result:
                for _, v in sample.items():
                    results.append(v['value'])
            return results
        else:
            return [result]
    
    def answer_generate_v0(self, question, **kwargs) -> List:
        prompt = self.prompter(question)
        answers_from_lm = self.lm.generate(prompt)
        generated_entities = []
        # numeric answers
        if answers_from_lm.strip().isnumeric():
            return [int(answers_from_lm)]
        # bool answers
        try: return [eval(answers_from_lm)]
        except: pass
        # string answers
        for line in answers_from_lm.strip().split('\n'):
            link_result = self.linker.link_entity_v1(line)
            if isinstance(link_result, str):
                generated_entities.append(link_result)
            elif isinstance(link_result, list) and len(link_result) != 0:
                generated_entities.append(link_result[0])
        return generated_entities
    
    def answer_generate_v1(self, sparql, question_type, **kwargs) -> List:
        return self.get_result(sparql)

    def answer_generate_v2(self, sparql, question_type, question, **kwargs) -> List:
        generated_entities = self.get_result(sparql)

        if question_type == '<yes or no>':
            prompt = self.yesorno_prompter(question)
            answer_from_lm = eval(self.lm.generate(prompt))

            if len(generated_entities) == 0:
                generated_entities = [answer_from_lm]
            else:
                generated_entities = [answer_from_lm | generated_entities[0]]
        return generated_entities

    def answer_generate_v3(self, sparql, question_type, question, **kwargs):
        generated_entities = self.get_result(sparql)

        if question_type == '<yes or no>':
            prompt = self.yesorno_prompter(question)
            answer_from_lm = eval(self.lm.generate(prompt))

            if len(generated_entities) == 0:
                generated_entities = [answer_from_lm]
            else:
                generated_entities = [answer_from_lm | generated_entities[0]]

        elif question_type == '<select>' and len(generated_entities) == 0:
            prompt = self.prompter(question)
            answers_from_lm = self.lm.generate(prompt)
            link_result = self.linker.link_entity_v1(answers_from_lm)
            if isinstance(link_result, str):
                generated_entities = [link_result]
        return generated_entities
    
    @log_time_cost("Answer Generation", reset=True)
    def answer_generate(self, **kwargs):
        return self.__getattribute__(f'answer_generate_{self.answer_generator_version}')(**kwargs)

if __name__ == '__main__':
    config = ConfigParser()
    config.read('conf/lc-quad-1.0.conf')
    g = AnswerGenerator(config)
    # print(g.answer_generate(
    #     answers_from_kb = False,
    #     question_type = '<yes or no>',
    #     question = 'Is Peter Piper Pizza in the pizza industry?'
    # ))
    print(g.answer_generate(
        sparql = 'SELECT DISTINCT ?TV_show WHERE {?TV_show <http://dbpedia.org/ontology/network> <http://dbpedia.org/resource/Prime_Time_Entertainment_Network>. ?TV_show <http://dbpedia.org/property/themeMusicComposer> <http://dbpedia.org/resource/Christopher_Franke>.}',
        question_type = '<select>',
        question = "Tell me the name of the Prime Time Entertainment Network's TV show whose Artist is Christopher Franke ?"
    ))