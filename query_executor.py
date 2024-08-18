'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
query_executor.py
'''
import SPARQLWrapper
import re
from SPARQLWrapper import JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError
from config import ConfigParser
from utils import log_time_cost


class QueryExecutor(object):
	def __init__(self, config: ConfigParser):
		self.iri = config.get('knowledge-base', 'iri')
		self.to_insert_iri = eval(config.get('knowledge-base', 'insert_iri'))
		self.sparql = config.get_sparql_engine()

	def is_sparql_correct(self, sparql: str):
		if sparql == '<no sparql>':
			return False
		try:
			self.sparql.setQuery(sparql)
			self.sparql.query().convert()
			return True
		except QueryBadFormed as e:
			return False

	def insert_iri(self, query: str):
		if self.iri:
			where_loc = query.upper().index("WHERE")
			return query[:where_loc] + f" FROM <{self.iri}> " + query[where_loc:]
		else:
			return query
	
	def preprocess(self, sparql):
		sparql = sparql.replace('ask ', 'ASK ').replace('select ', 'SELECT ').replace('prefix', 'PREFIX')
		if 'SELECT' in sparql:
			query_type = 'SELECT'
		elif 'ASK' in sparql:
			query_type = 'ASK'
		else:
			raise NotImplementedError('Unsupported query:' + sparql)
		index = sparql.index('ASK') if query_type == 'ASK' else sparql.index('SELECT')
		prefix, body = sparql[:index], sparql[index:]

		for pre, uri in re.findall(r'PREFIX\s+(.+?)\s?:\s?<([^<>]+?)>', prefix):
			for to_remove in re.findall(f'{pre}:([^\s\.\(,\)]+)', body):
				body = body.replace(f'{pre}:{to_remove}', f'<{uri}{to_remove}>')
		return body

	@log_time_cost('KB Endpoint execution')
	def execute(self, query) :
		if self.to_insert_iri:
			query = self.insert_iri(query)
		# print(query)
		query = self.preprocess(query)
		# print(query)
		self.sparql.setQuery(query)
		try:
			result = self.sparql.queryAndConvert()
		except EndPointInternalError as e:
			return []

		if self.sparql.queryType == 'ASK':
			return result['boolean']
		
		if self.sparql.queryType == 'SELECT':
			return result['results']['bindings']
		
		# execute query with endpoint engine
		return result

if __name__ == '__main__':
	config = ConfigParser()
	config.read('conf/qald-9.conf')
	query = 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX dbc: <http://dbpedia.org/resource/Category:> PREFIX dct: <http://purl.org/dc/terms/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> SELECT DISTINCT ?s WHERE { ?s dct:subject dbc:Missions_to_the_Sun ; rdf:type ?type FILTER ( ?type IN (dbo:Satellite, dbo:ArtificialSatellite) ) }'
	sparql = QueryExecutor(config)
	print(sparql.execute(query))
