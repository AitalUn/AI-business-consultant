import getpass
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from haystack import Document, Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma.retriever import ChromaEmbeddingRetriever
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers import DocumentWriter


llm_model = 'gpt-4o-mini'
embedding_model = 'text-embedding-3-large'
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

embedder = OpenAIDocumentEmbedder(model=embedding_model)
llm = OpenAIGenerator(model=llm_model)

document_store = ChromaDocumentStore(host="localhost", port="8000")
writer = DocumentWriter(document_store)

root = Path(__file__).parent
df = pd.read_csv(root / "tmp/labelled_newscatcher_dataset.csv", sep = ';')
df['published_date'] = pd.to_datetime(df['published_date'])
current_date = df.loc[df['published_date'].idxmax()]['published_date']
one_day_ago = current_date - timedelta(days=1)
recent_df = df[df['published_date'] >= one_day_ago]
recent_df_reduced = recent_df.sample(n = 100, random_state = 42) #для теста так пойдёт
print(recent_df.size)

docs = [Document(content=x['title'], meta={'link': x['link']}) for _, x in recent_df_reduced.iterrows()]
print(len(docs))


indexing = Pipeline()
indexing.add_component("embedder", embedder)
indexing.add_component("writer", writer)
indexing.connect("embedder", "writer")

indexing.run({"embedder": {"documents": docs}})


retriever = ChromaEmbeddingRetriever(document_store=document_store)
text_embedder = OpenAITextEmbedder(model= embedding_model)

retriever_pipe = Pipeline()
retriever_pipe.add_component("embedder", text_embedder)
retriever_pipe.add_component("retriever", retriever)
retriever_pipe.connect("embedder.embedding", "retriever.query_embedding")

def retrieve(query):
  answer = retriever_pipe.run({"embedder": {"text": query}})
  return answer['retriever']['documents']


print(retrieve("Oil"))