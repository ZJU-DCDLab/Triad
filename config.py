import configparser
from elasticsearch import Elasticsearch
from SPARQLWrapper import SPARQLWrapper, JSON
from lm import OpenAI_Language_Model, HuggingFaceModel
from utils import log_time_cost


class ConfigParser(configparser.ConfigParser):
    def read(self, *args, **kwargs):
        super().read(*args, **kwargs)
        self.lm_openai = OpenAI_Language_Model(self)
        print('done!')
        self.es = Elasticsearch(
            hosts=self.get('elastic-search', 'endpoint'), 
            timeout=120, 
            max_retries=10, 
            retry_on_timeout=True
        )
        self.sparql = SPARQLWrapper(self.get('knowledge-base', 'endpoint'))
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(300)
        
    def get_lm(self, model='openai'):
        if model == 'openai':
            return self.lm_openai
        elif model == 'huggingface':
            return self.lm_hf

    def get_sparql_engine(self):
        return self.sparql

    def get_es(self):
        self.es.search = log_time_cost('ES Endpoint search')(self.es.search)
        return self.es

if __name__ == '__main__':
    conf = ConfigParser()
    conf.read('conf/lc-quad-1.0.conf')
