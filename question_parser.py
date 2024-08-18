'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
question_parser.py
'''
from prompter import Prompter
import re
from config import ConfigParser
from utils import log_time_cost
from lm import OpenAI_Language_Model


class QuestionParseResult:
	def __init__(self, *, question) -> None:
		self.question = question
		self.task_type = None

		self.variants = dict()
		self._variants = dict()
	
	def set_task_type(self, task_type):
		self.task_type = task_type

	def set_triples(self, triples):
		self._triples = self._extract_triples(triples)
		self.triples = []
		for triple in self._triples:
			s, p, o = triple
			s = '_'.join(s.split()) if s.startswith('?') else s
			p = '_'.join(p.split()) if p.startswith('?') else p
			o = '_'.join(o.split()) if o.startswith('?') else o
			self.triples.append(f'<{s}, {p}, {o}>')

	def set_variant(self, triple, variants):
		self._variants[self._extract_triple(triple)] = set(self._extract_triples(variants))
		self.variants = {f'<{k[0]}, {k[1]}, {k[2]}>': set([f'<{item[0]}, {item[1]}, {item[2]}>' for item in v]) for k,v in self._variants.items()}

	def _extract_triple(self, triple):
		return re.search(r'<\s?(.+?),\s?(.+?),\s?(.+?)>', triple).groups()

	def _extract_triples(self, triples):
		return list(re.findall(r'<\s?(.+?),\s?(.+?),\s?(.+?)>', triples))
				
	def __str__(self):
		if len(self.variants) > 0:
			return "<QuestionParseResult>(\n  question={}, \n  triples={}, \n  task_type={}, \n  variants={})".format(
					self.question, self.triples, self.task_type, self.variants)
		else:
			return "<QuestionParseResult>(\n  question={}, \n  triples={}, \n  task_type={})".format(
					self.question, self.triples, self.task_type)
				
	def get_triples(self):
		return self._triples

	def get_variants(self, triple):
		return self._variants[triple]
	

class QuestionParser(object):
	def __init__(self, config: ConfigParser):
		self.subtask_list = eval(config.get('question-parse', 'subtask_list'))
		assert isinstance(self.subtask_list, list)
		self.prompters = { 
			sub_task: Prompter(
				task=sub_task, 
				shots=eval(config.get('question-parse', 'few_shots')), 
				cot=eval(config.get('question-parse', 'cot'))
			) for sub_task in self.subtask_list}
		self.lm = config.get_lm(config.get('question-parse', 'model'))

	@log_time_cost("Question Parsing", reset=True)
	def parse(self, question):
		parse_result = QuestionParseResult(question=question)
		for subtask in self.subtask_list:
			if subtask == 'var_iden':
				for triple in parse_result.triples:
					prompt = self.preprocess(question, subtask)
					prompt = prompt.replace('<response_from_previous_task>', triple)
					res = self.lm.generate(prompt, decode_fn=self.prompters[subtask].decode_output)
					parse_result.set_variant(triple, res)
			else:
				prompt = self.preprocess(question, subtask)
				res = self.lm.generate(prompt, decode_fn=self.prompters[subtask].decode_output)
				if subtask == 'triple_iden':
					parse_result.set_triples(res)
				elif subtask == 'ans_cls':
					parse_result.set_task_type(res)
		return parse_result

	def preprocess(self, question, subtask):
		return self.prompters[subtask](question)

	def postprocess(self, output, subtask):
		pass


if __name__ == '__main__':
	config = ConfigParser()
	config.read('conf/lc-quad-1.0.conf')
	parser = QuestionParser(config)
	question = "What currencies are used in the country whose capital is Podgorica?"
	print(parser.parse(question))
