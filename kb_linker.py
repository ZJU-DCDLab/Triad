'''
Copyright (c) 2023 OpenKS Authors, DCD Research Lab, Zhejiang University.
All Rights Reserved.
kb_linker.py
'''
from query_executor import QueryExecutor
from elasticsearch import helpers
from urllib.parse import unquote
from tqdm import tqdm
from strsimpy.levenshtein import Levenshtein
from prompter import Prompter
from typing import Set, List, Tuple, Union
from config import ConfigParser
from itertools import product
import os
import json
from utils import log_time_cost


class KB_Linker(object):
    def __init__(self, config: ConfigParser) -> None:
        self.kb = config.get('knowledge-base', 'kb')
        self.version = config.get('knowledge-base', 'version')
        self.entity_linking_version = config.get('knowledge-base-linking', 'entity_linking_version')
        self.predicates_linking_version = config.get('knowledge-base-linking', 'predicates_linking_version')
        self.index = config.get('elastic-search', 'index')

        self.es = config.get_es()
        self.lm = config.get_lm(config.get('knowledge-base-linking', 'model'))
        self.query_executor = QueryExecutor(config)

        self.levenshtein = Levenshtein()
        for version in ['v1', 'v2', 'v3']:
            self.__setattr__(f"entity_linking_prompter_{version}", 
                             Prompter(task=f'entity_linking_{version}'))
        self.predicate_linking_prompter = Prompter(task='predicate_linking')
        
        with open('index_mapping.json', 'r') as f:
            self.mappings = json.load(f)
        
        self.config = config
        self.inplicit_linking = eval(config.get('knowledge-base-linking', 'inplicit_linking'))
        self.numbers_to_text = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}

        if not self.es.indices.exists(index=self.index):
            self._create_index()
    
    def _get_config_or_default(self, name, default):
        if name in [k for k, _ in self.config.items('knowledge-base-linking')]:
            return eval(self.config.get('knowledge-base-linking', name))
        else:
            return default
        
    def _extract_semantic_info(self, uri):
        sparql = f"SELECT ?name WHERE {{ <{uri}>  <http://www.w3.org/2000/01/rdf-schema#label> ?name}}"
        response = self.query_executor.execute(sparql)
        try:
            if len(response) == 1:
                res = response[0]
                return res['name']['value']
            elif len(response) > 1:
                for res in response:
                    if res['name']['xml:lang'] == 'en':
                        return res['name']['value']
        except:
            pass
        return self._split_camel_case(unquote(uri.split('/')[-1]).replace('_', ' '))

    def _entity_generator(self):
        sparql = "SELECT DISTINCT ?s ?o WHERE {?s <http://www.w3.org/2000/01/rdf-schema#label> ?o}"
        response = self.query_executor.execute(sparql)
        for res in response:
            yield res['s']['value'], res['o']['value']
    
    def _relation_generator(self):
        sparql = "SELECT DISTINCT ?p WHERE {?s ?p ?o}"
        response = self.query_executor.execute(sparql)
        for res in response:
            yield res['p']['value'], self._extract_semantic_info(res['p']['value'])

    def _create_index(self):
        if self.es.indices.exists(index=self.index):
            ensure = input('Index already exists, are you sure to delete? (y/n):')
            if ensure.strip() == 'y':
                self.es.indices.delete(index=self.index)
            else:
                print('Failed to create index.')
                return
        self.es.indices.create(index=self.index, mappings=self.mappings)

        action_list = []
        for uri, txt in self._relation_generator():
            action_list.append({
                'type': 'relation', 
                'text': txt,
                'uri': uri
            })
        print("Load Relations: ", len(action_list))
        helpers.bulk(self.es, action_list, index=self.index)

        
        with open(os.path.join('kb', self.kb, self.version, 'labels.ttl'), 'r') as f:
            action_list = []
            for line in tqdm(f):
                try:
                    uri, text = line.strip().split('\t', maxsplit=1)
                except:
                    print(line)
                action_list.append({
                    'type': 'entity', 
                    'text': text,
                    'uri': uri
                })
                # reduce memory usage
                if len(action_list) % 10000 == 0:
                    helpers.bulk(self.es, action_list, index=self.index)
                    action_list = []
            if action_list:
                helpers.bulk(self.es, action_list, index=self.index)

    def _split_camel_case(self, string):
        tokens = []
        token = []
        for prev, char, next in zip(' ' + string, string, string[1:] + ' '):
            if self._is_camel_case_boundary(prev, char, next):
                if token:
                    tokens.append(''.join(token))
                token = [char]
            else:
                token.append(char)
        if token:
            tokens.append(''.join(token))
        return ' '.join(tokens)

    def _is_camel_case_boundary(self,prev, char, next):
        if prev.isdigit():
            return not char.isdigit()
        if char.isupper():
            return next.islower() or prev.isalpha() and not prev.isupper()
        return char.isdigit()
    
    def is_kb_related(self, uri: str) -> bool:
        if self.kb == 'dbpedia':
            blacklist = [
                'http://dbpedia.org/ontology/wikiPageRedirects',
                'http://dbpedia.org/ontology/wikiPageRevisionID',
                'http://dbpedia.org/ontology/wikiPageID',
                'http://dbpedia.org/ontology/wikiPageDisambiguates',
                'http://dbpedia.org/ontology/wikiPageExternalLink'
            ]
        else:
            blacklist = []
        return uri not in blacklist
    
    def link_entity_v1(self, entity: str, **kwargs) -> str:
        entity_topk = self._get_config_or_default('entity_topk', 2)
        if entity.startswith('?'):
            return '_'.join(entity.split())
        query = {
            "bool": {
                "must": [
                    {
                        "match": { 
                            "text": entity
                        }
                    },{
                        "term": { 
                            "type": "entity"
                        }
                    }
                ]
            }
        }
        results = self.es.search(index=self.index, query=query, size=10)['hits']['hits']
        if len(results) == 0:
            return []
        uri_words_mapping = { 
            result['_source']['uri']: result['_source']['text'].replace('  ', ' ')
            for result in results
        }
        similarities = [-self.levenshtein.distance(entity, v) for _, v in uri_words_mapping.items()]
        uris = list(uri_words_mapping.keys())
        temp = sorted(zip(uris, similarities), key=lambda x: x[1], reverse=True)
        return [uri for uri,_ in temp[:entity_topk]]

    def link_entity_v2(self, entity: str, sentence: str, **kwargs) -> str:
        topk = self._get_config_or_default('topk-ent', 10)
        entity_topk = self._get_config_or_default('entity_topk', 2)
        if entity.startswith('?'):
            return '_'.join(entity.split())
        query = {
            "bool": {
                "must": [
                    {
                        "match": { 
                            "text": entity
                        }
                    },{
                        "term": { 
                            "type": "entity"
                        }
                    }
                ]
            }
        }
        results = self.es.search(index=self.index, query=query, size=topk)['hits']['hits']
        results = [result['_source']['uri'] for result in results ]

        prompt = self.entity_linking_prompter_v2(
            self.numbers_to_text[entity_topk], 
            self.numbers_to_text[entity_topk], 
            sentence, 
            entity, 
            '\n  '+'\n  '.join(results))
        result = [r.strip() for r in self.lm.generate(prompt).split('\n')]
        return result
    
    def link_entity_v3(self, entity: str, sentence: str, **kwargs) -> str:
        topk = self._get_config_or_default('topk-ent', 10)
        entity_topk = self._get_config_or_default('entity_topk', 2)
        if entity.startswith('?'):
            return '_'.join(entity.split())
        query = {
            "bool": {
                "must": [
                    {
                        "match": { 
                            "text": entity
                        }
                    },{
                        "term": { 
                            "type": "entity"
                        }
                    }
                ]
            }
        }
        results = self.es.search(index=self.index, query=query, size=topk*10)['hits']['hits']
        results = [result['_source']['uri'] for result in results ]


        if self.kb == 'dbpedia':
            query = 'SELECT ?abstract WHERE {{ <{}> <http://dbpedia.org/ontology/abstract> ?abstract}}'
        abstracts = {}
        for result in results:
            try:
                res = self.query_executor.execute(query.format(result))
            except Exception as e:
                print(e)
                continue
            if res:
                res = res[0]['abstract']['value']
                abstracts[result] = res.split('.')[0]
                if len(abstracts) == topk:
                    break
        prompt = self.entity_linking_prompter_v3(
            self.numbers_to_text[entity_topk], 
            self.numbers_to_text[entity_topk], 
            sentence,
            entity, 
            '\n  '+'\n  '.join(['{} : {}'.format(uri, abstract) for uri, abstract in abstracts.items()])
        )
        result = self.lm.generate(prompt)
        result = [r.strip() for r in self.lm.generate(prompt).split('\n')]
        return result
    
    def link_predicate_v1(self, predicate: str, s=None, o=None, return_score: bool=False, only_ontology=False, **kwargs) -> Union[List[str], List[Tuple[str,str,str]]]:
        if only_ontology:
            query = {
            "bool": {
                "must": [
                    {
                        "match": { 
                            "text": predicate
                        }
                    },{
                        "term": { 
                            "type": "relation"
                        }
                    },{
                        "match": {
                            "uri": "ontology"
                        }
                    }
                ]
            }
        }
        else:
            query = {
                "bool": {
                    "must": [
                        {
                            "match": { 
                                "text": predicate
                            }
                        },{
                            "term": { 
                                "type": "relation"
                            }
                        }
                    ]
                }
            }
        results = self.es.search(index=self.index, query=query, size=20)['hits']['hits']
        if len(results) == 0:
            final_result = []
        else:
            max_score = max([result['_score'] for result in results])
            uri_words_mapping = { 
                result['_source']['uri']:result['_source']['text'].replace('  ', ' ')
                for result in results 
                if result['_score'] == max_score 
            }
            similarities = [-self.levenshtein.distance(predicate, v) for _, v in uri_words_mapping.items()]
            max_similarity = max(similarities)
            final_result = []
            for w, similarity in zip(uri_words_mapping.keys(), similarities):
                if similarity == max_similarity:
                    final_result.append(w)
        if return_score:
            return (final_result, max_similarity) if s is None else ([(s, p, o) for p in final_result], max_similarity)
        return final_result if s is None else [(s, p, o) for p in final_result]
    
    def link_predicate_v2(self, s: str, o: str, sentence: str, predicate=None, **kwargs) -> List[Tuple[str,str,str]]:
        if s.startswith('?') and o.startswith('?'):
            return self.link_predicate_v1(predicate, s, o)
        s_uri = f'<{s}>' if not s.startswith('?') else s
        o_uri = f'<{o}>' if not o.startswith('?') else o

        candidates_predicates = set()
        # give s and o, find possible candidates p
        sparql = f'SELECT ?p WHERE {{ {s_uri} ?p {o_uri} .}}'
        res = self.query_executor.execute(sparql)
        for instance in res:
            candidates_predicates.add(instance['p']['value'])
        # maybe s and o switch the order
        sparql =  f'SELECT ?p WHERE {{ {o_uri} ?p {s_uri} .}}'
        res = self.query_executor.execute(sparql)
        for instance in res:
            candidates_predicates.add(instance['p']['value'])

        candidates_predicates = set(candidates_predicates)
        candidates_predicates = [p for p in candidates_predicates if self.is_kb_related(p)]
        
        prompt = self.predicate_linking_prompter('one', 'one', sentence, f'\n  {s}\n  {o}' ,'  '+'\n  '.join(candidates_predicates))
        query_result = self.lm.generate(prompt)

        # choose order
        sparql = 'SELECT 1 WHERE { %s %s %s .}' %(s_uri, f'<{query_result}>' , o_uri)
        res = self.query_executor.execute(sparql)
        if len(res) != 0:
            return [(s, query_result, o)]
        else:
            return [(o, query_result, s)]
        
    def link_predicate_v3(self, s: str, o: str, sentence: str, predicate=None, **kwargs) -> List[Tuple[str,str,str]]:
        topk = self._get_config_or_default('predicate_topk', 2)
        if s.startswith('?') and o.startswith('?'):
            return self.link_predicate_v1(predicate, s, o)
        s_uri = f'<{s}>' if not s.startswith('?') else s
        o_uri = f'<{o}>' if not o.startswith('?') else o
        # give s and o, find possible candidates p
        candidates_predicates = set()
        sparql = f'SELECT ?p WHERE {{ {s_uri} ?p {o_uri} .}}'
        res = self.query_executor.execute(sparql)
        for instance in res:
            uri = instance['p']['value']
            if self.is_kb_related(uri):
                candidates_predicates.add(uri)
        if len(candidates_predicates) == 0:
            return []

        prompt = self.predicate_linking_prompter(
            self.numbers_to_text[topk], 
            self.numbers_to_text[topk][topk], 
            sentence, 
            f'\n  {s_uri}\n  {o_uri}' ,'  '+'\n  '.join(sorted(candidates_predicates))
        )
        query_result = self.lm.generate(prompt)
        results = query_result.split('\n')
        return [(s, result.strip(), o) for result in results]

    def link_predicate_v4(self, s: str, o: str, sentence: str, predicate=None, **kwargs) -> Set[str]:
        p = self._get_config_or_default('p', 10)
        if s.startswith('?') and o.startswith('?'):
            return self.link_predicate_v1(predicate, s, o)
        s_uri = f'<{s}>' if not s.startswith('?') else s
        o_uri = f'<{o}>' if not o.startswith('?') else o

        # give s and o, find possible candidates p
        candidates_predicates = set()
        sparql = f'SELECT ?p WHERE {{ {s_uri} ?p {o_uri} .}}'
        res = self.query_executor.execute(sparql)
        for instance in res:
            uri = instance['p']['value']
            if self.is_kb_related(uri):
                candidates_predicates.add(uri)

        if len(candidates_predicates) == 0:
            return []
        
        candidates_predicates = list(candidates_predicates)

        candidates_predicates_v2 = []
        for i in range(0, len(candidates_predicates), p):
            candidates = candidates_predicates[i:i+p]
            prompt = self.predicate_linking_prompter('one', 'one', sentence, f'\n  {s_uri}\n  {o_uri}' ,'  '+'\n  '.join(candidates))
            query_result = self.lm.generate(prompt)
            results = [result.strip() for result in query_result.split('\n') if self.is_kb_related(result)]
            candidates_predicates_v2.extend(results)

        # step 2
        prompt = self.predicate_linking_prompter('one', 'one', sentence, f'\n  {s_uri}\n  {o_uri}' ,'  '+'\n  '.join(candidates_predicates_v2))
        query_result = self.lm.generate(prompt)
        results = query_result.split('\n')
        
        return [(s, result.strip(), o) for result in results]
    
    def link_predicate_v5(self, s: str, o: str, sentence: str, predicate=None, **kwargs) -> Set[str]:
        topk = self._get_config_or_default('predicate_topk', 2)
        if s.startswith('?') and o.startswith('?'):
            return self.link_predicate_v1(predicate, s, o)
        s_uri = f'<{s}>' if not s.startswith('?') else s
        o_uri = f'<{o}>' if not o.startswith('?') else o

        # give s and o, find possible candidates p
        candidates_predicates = set()
        sparql = f'SELECT ?p WHERE {{ {s_uri} ?p {o_uri} .}}'
        res = self.query_executor.execute(sparql)
        for instance in res:
            uri = instance['p']['value']
            if self.is_kb_related(uri):
                candidates_predicates.add(uri)

        if len(candidates_predicates) == 0:
            return []
        
        candidates_predicates = list(candidates_predicates)

        candidates = []
        for _ in range(topk):
            prompt = self.predicate_linking_prompter('one', 'one', sentence, f'\n  {s_uri}\n  {o_uri}' ,'  '+'\n  '.join(candidates_predicates))
            query_result = self.lm.generate(prompt).split('\n')[0].strip()
            if query_result not in candidates_predicates:
                continue
            candidates.append(query_result)
            candidates_predicates.remove(query_result)
        return [(s, result, o) for result in candidates]
    
    def link_predicate_v6(self, s: str, o: str, sentence: str, predicate=None, previous_triples=[], **kwargs) -> Set[str]:
        topk = self._get_config_or_default('predicate_topk', 2)
        candidates_predicates = set()
        if s.startswith('?') and o.startswith('?'):
            for linked_triples in product(*previous_triples):
                format_triple = ''
                for linked_s, linked_p, linked_o in linked_triples:
                    linked_s = f'<{linked_s}>' if not linked_s.startswith('?') else linked_s
                    linked_p = f'<{linked_p}>' if not linked_p.startswith('?') else linked_p
                    linked_o = f'<{linked_o}>' if not linked_o.startswith('?') else linked_o
                    format_triple += f"{linked_s} {linked_p} {linked_o}."
                sparql = f'SELECT ?p WHERE {{ {format_triple} {s} ?p {o} .}}'
                res = self.query_executor.execute(sparql)
                for instance in res:
                    uri = instance['p']['value']
                    if self.is_kb_related(uri):
                        candidates_predicates.add(uri)  
        else:
            s_uri = f'<{s}>' if not s.startswith('?') else s
            o_uri = f'<{o}>' if not o.startswith('?') else o

            # give s and o, find possible candidates p
            sparql = f'SELECT ?p WHERE {{ {s_uri} ?p {o_uri} .}}'
            res = self.query_executor.execute(sparql)
            for instance in res:
                uri = instance['p']['value']
                if self.is_kb_related(uri):
                    candidates_predicates.add(uri)
            if len(candidates_predicates) == 0:
                return []
        candidates_predicates = list(candidates_predicates)
        prompt = self.predicate_linking_prompter(
            self.numbers_to_text[topk], 
            self.numbers_to_text[topk], 
            sentence, 
            f'\n  {s}\n  {o}' ,'  '+'\n  '.join(sorted(candidates_predicates))
        )
        query_result = self.lm.generate(prompt)
        results = query_result.split('\n')
        return [(s, result.strip(), o) for result in results]
    
    def link(self, triple, **kwargs) -> List[Tuple[str,str,str]]:
        s, p, o = triple
        s = self.__getattribute__(f'link_entity_{self.entity_linking_version}')(entity=s, **kwargs)
        o = self.__getattribute__(f'link_entity_{self.entity_linking_version}')(entity=o, **kwargs)

        if not isinstance(s, list):
            s = [s]
        if not isinstance(o, list):
            o = [o]

        candidates_triples = set()
        for ss, oo in product(s, o):
            # give s and o, find possible candidates p 
            triples = self.__getattribute__(f'link_predicate_{self.predicates_linking_version}')(s=ss, o=oo, predicate=p, **kwargs)
            candidates_triples = candidates_triples | set(triples)
            # maybe s and o switch the order
            triples = self.__getattribute__(f'link_predicate_{self.predicates_linking_version}')(s=oo, o=ss, predicate=p, **kwargs)
            candidates_triples = candidates_triples | set(triples)

        return list(candidates_triples)   

    def _link_implicit(self, implicit_entity: str) -> List:
        assert implicit_entity.startswith('?')
        ontology = implicit_entity[1:]
        a = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
        res = self.link_predicate_v1(ontology, only_ontology=True)
        return [(implicit_entity, a, r) for r in res]

    @log_time_cost('KB Linker', reset=True)
    def link_all(self, triples, **kwargs) -> List[List[Tuple[str,str,str]]]:
        # implicit_entities = set()
        # for triple in triples:
        #     s, p, o = triple
        #     if s.startswith('?'):
        #         implicit_entities.add(s)
        #     if o.startswith('?'):
        #         implicit_entities.add(o)

        all_candidate_triples = [] # List[List]
        for triple in triples:
            candidates_triples = set()
            s, p, o = triple
            s = self.__getattribute__(f'link_entity_{self.entity_linking_version}')(entity=s, **kwargs)
            o = self.__getattribute__(f'link_entity_{self.entity_linking_version}')(entity=o, **kwargs)

            if not isinstance(s, list):
                s = [s]
            if not isinstance(o, list):
                o = [o]
            for ss, oo in product(s, o):
                # give s and o, find possible candidates p 
                triples = self.__getattribute__(f'link_predicate_{self.predicates_linking_version}')(s=ss, o=oo, predicate=p, previous_triples=all_candidate_triples, **kwargs)
                candidates_triples = candidates_triples | set(triples)
                # maybe s and o switch the order
                triples = self.__getattribute__(f'link_predicate_{self.predicates_linking_version}')(s=oo, o=ss, predicate=p, previous_triples=all_candidate_triples, **kwargs)
                candidates_triples = candidates_triples | set(triples)
            all_candidate_triples.append(candidates_triples)
        
        # extend implicit entities
        # if self.inplicit_linking:
        #     for implicit_entity in implicit_entities:
        #         all_candidate_triples.append([('','','')] + self._link_implicit(implicit_entity))
        return all_candidate_triples


if __name__ == '__main__':
    config = ConfigParser()
    config.read('conf/lc-quad-1.0.conf')
    linker = KB_Linker(config)
    linker._create_index()
