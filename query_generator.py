'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
query_generator.py
'''
from query_executor import QueryExecutor
from question_parser import QuestionParseResult
from kb_linker import KB_Linker
from prompter import Prompter
from lm import OpenAI_Language_Model
from typing import List
from itertools import product
from config import ConfigParser
from utils import log_time_cost
import re


def format_uri(triple):
	s, p, o = triple
	s = s if s.startswith('?') else f'<{s}>'
	p = p if p.startswith('?') else f'<{p}>'
	o = o if o.startswith('?') else f'<{o}>'
	return f' {s} {p} {o}. '

def rank_triples(triple):
	s, p, o = triple
	if s.startswith('?') and o.startswith('?'):
		return 2
	elif s.startswith('?') or o.startswith('?'):
		return 1
	else:
		return 0
	
class QueryGenerator(object):
	def __init__(self, config: ConfigParser):
		self.sparql = QueryExecutor(config)
		self.lm = config.get_lm(config.get('sparql-generation', 'model'))
		self.linker = KB_Linker(config)
		self.sparql_generation_version = config.get('sparql-generation', 'sparql_generation_version')
		self.prompter = Prompter(
			task=f'sparql_{self.sparql_generation_version}', 
			shots=eval(config.get('sparql-generation', 'few_shots')), 
			cot=eval(config.get('sparql-generation', 'cot'))
		)
		self.sparql_choose_prompter = Prompter(task='sparql_choose')

	def extract_triples_from_sparql(self, sparql: str) -> List[tuple]:
		where_loc = sparql.upper().index('WHERE')
		sparql = sparql[where_loc+len("WHERE"): ]
		pattern = r'(?:<.+?>|\?.+?)\s+(?:<.+?>|\?.+?)\s+(?:<.+?>|\?.+?)[\s\.]+'
		raw = re.findall(pattern, sparql)
		pattern = r'(<.+?>|\?.+?)\s+(<.+?>|\?.+?)\s+(<.+?>|\?.+?)[\s\.]+'
		triples = [re.findall(pattern, trip)[0] for trip in raw]
		temp = sorted(zip(raw, triples), key=lambda x: rank_triples(x[1]))
		raw = [t[0] for t in temp]
		triples = [t[1] for t in temp]
		return raw, triples

	def generate_v1(self, input: QuestionParseResult, return_prompt=False, **kwargs):
		def generate_prompt(question, question_type, triples_in_uris):
			triples_in_uris = [f'<{triple[0]}, {triple[1]}, {triple[2]}>' for triple in triples_in_uris]
			prompt = self.prompter(question, question_type, ' '.join(triples_in_uris))
			return prompt
		triples = input.get_triples()
		triples_in_uris = [
			self.linker.link(triple=triple, sentence=input.question) # List[Tuple[str]]
		for triple in triples] 
		# to make SPARQL query generation prompt using input
		query_gen_prompts = [
			generate_prompt(input.question, input.task_type, item)
			for item in product(*triples_in_uris)
		]
		if len(query_gen_prompts) == 0:
			return ('<no sparql>', '<no prompt>') if return_prompt else '<no sparql>'
		for query_gen_prompt in query_gen_prompts:
			sparql = self.lm.generate(query_gen_prompt, self.prompter.decode_output) # sparql
			temp_result = self.sparql.execute(sparql)
			if input.task_type == '<yes or no>' and temp_result == True:
				break
			if input.task_type != '<yes or no>' and len(temp_result) != 0:
				break
		return (sparql, query_gen_prompt) if return_prompt else sparql

	def generate_v2(self, input: QuestionParseResult, return_prompt=False, **kwarg):
		query_gen_prompt = self.prompter(input.question, input.task_type, ' '.join(input.triples))
		sparql_template = self.lm.generate(query_gen_prompt, self.prompter.decode_output).replace('\n', ' ') # sparql
		raws, triples = self.extract_triples_from_sparql(sparql_template) # List[Tuple[str]]
		triples_in_uri = self.linker.link_all(triples=triples, sentence=input.question) # List[List[Tuple[str]]]
	
		sparqls = []
		for item in product(*triples_in_uri):
			sparql = sparql_template
			for raw, linked in zip(raws, item):
				sparql = sparql.replace(raw, format_uri(linked))
			sparqls.append(sparql)
		
		if len(sparqls) == 0:
			return ('<no sparql>', '<no prompt>') if return_prompt else '<no sparql>'
		for sparql in sparqls:
			temp_result = self.sparql.execute(sparql)
			if input.task_type == '<yes or no>' and temp_result == True:
				break
			if input.task_type != '<yes or no>' and len(temp_result) != 0:
				break
		return (sparql, query_gen_prompt) if return_prompt else sparql

	def generate_v3(self, input: QuestionParseResult, return_prompt=False, **kwarg):
		query_gen_prompt = self.prompter(input.question, input.task_type, ' '.join(input.triples))
		sparql_template = self.lm.generate(query_gen_prompt, self.prompter.decode_output).replace('\n', ' ') # sparql
		raws, triples = self.extract_triples_from_sparql(sparql_template) # List[Tuple[str]]
		triples_in_uri = self.linker.link_all(triples=triples, sentence=input.question) # List[List[Tuple[str]]]
	
		sparqls = []
		for item in product(*triples_in_uri):
			sparql = sparql_template
			for raw, linked in zip(raws, item):
				sparql = sparql.replace(raw, format_uri(linked))
			sparqls.append(sparql)
		
		if len(sparqls) == 0:
			return ('<no sparql>', '<no prompt>') if return_prompt else '<no sparql>'
		for sparql in sparqls:
			temp_result = self.sparql.execute(sparql)
			if input.task_type == '<yes or no>' and temp_result == True:
				break
			if input.task_type != '<yes or no>' and len(temp_result) != 0:
				break
		return (sparql, query_gen_prompt) if return_prompt else sparql

	def generate_v4(self, input: QuestionParseResult, return_prompt=False, return_intermediate=False, **kwarg):
		query_gen_prompt = self.prompter(input.question, input.task_type, ' '.join(input.triples))
		sparql_template = self.lm.generate(query_gen_prompt, self.prompter.decode_output).replace('\n', ' ') # sparql
		raws, triples = self.extract_triples_from_sparql(sparql_template) # List[Tuple[str]]
		triples_in_uri = self.linker.link_all(triples=triples, sentence=input.question) # List[List[Tuple[str]]]

		sparqls = []
		for item in product(*triples_in_uri):
			sparql = sparql_template
			for raw, linked in zip(raws, item):
				sparql = sparql.replace(raw, format_uri(linked))
			sparqls.append(sparql)
			
		candidates = []
		for sparql in sparqls:
			temp_result = self.sparql.execute(sparql)
			if isinstance(temp_result, bool) or len(temp_result) != 0:
				candidates.append(sparql)
		
		if len(candidates) == 0:
			return ('<no sparql>', '<no prompt>') if return_prompt else '<no sparql>'
		
		if len(candidates) == 1:
			sparql = candidates[0]
		else:
			prompt = self.sparql_choose_prompter(input.question, '  '+'\n  '.join(candidates))
			sparql = self.lm.generate(prompt)
		if return_intermediate:
			return sparql, triples_in_uri, sparql_template
		return (sparql, query_gen_prompt) if return_prompt else sparql

	@log_time_cost("Query Generation", reset=True)
	def generate(self, **kwargs):
		return self.__getattribute__(f"generate_{self.sparql_generation_version}")(**kwargs)
		
if __name__ == '__main__':
	config = ConfigParser()	
	config.read('conf/lc-quad-1.0.conf')
	qg = QueryGenerator(config)
	# print(qg.extract_triples_from_sparql('SELECT DISTINCT ?architect WHERE { ?architect <tenant> <New Sanno hotel>. ?architect <architect_of> <Marine Corps Air Station Kaneohe Bay>. }'))
	from question_parser import QuestionParser
	parser = QuestionParser(config)
	# question = "Which sects people are part of local political parties which fall under the International Muslim Brotherhood?"
	# question = "Is Peter Piper Pizza in the pizza industry?"
	
	# question = "What is the total number of other tenant of the stadia whose one of the tenant is Raptors 905?"
	question = "What currencies are used in the country whose capital is Podgorica?"
	# question = "Which U.S. state has been admitted latest?"
	res = parser.parse(question)
	print(res)
	print(qg.generate(input=res))
