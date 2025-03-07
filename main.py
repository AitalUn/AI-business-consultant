from haystack import Document, Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore(hosts = "http://localhost:9200")