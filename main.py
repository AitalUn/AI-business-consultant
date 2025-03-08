import streamlit as st
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
import os

# Получаем API-ключ OpenAI из переменной окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Подключение к ChromaDB
database = ChromaDocumentStore(host="localhost", port="8000")

# Настройка компонентов
retriever = ChromaEmbeddingRetriever(document_store=database)
embedder = OpenAITextEmbedder(model="text-embedding-3-large")
generator = OpenAIGenerator(model= "gpt-4o-mini")

# Создание RAG пайплайна
prompt = """
Ты помощник для бизнеса, тебе надо посмотреть на названия новостей и по заголовкам 
орпеделить какая сейчас ситуация связанная с его бизнесом, а также предложить методы решения проблемы

Проблема клиента: {{query}}

сайты:
{% for document in documents %}
    {{ document.content }}
{% endfor %}
"""

prompt_builder = PromptBuilder(template=prompt)

rag = Pipeline()
rag.add_component("embedder", embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt", prompt_builder)
rag.add_component("generator", generator)

rag.connect("embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt", "generator")



# print("*"*100, "\n", rag.run({"embedder": {"text": "Oil"}}))

# Интерфейс в Streamlit
st.set_page_config(page_title="Business RAG", layout="wide")
st.title("AI-Помощник для бизнеса")

query = st.text_input("Введите ваш запрос:", "")

if query:
    with st.spinner("Ищем данные..."):
        answer = rag.run({"embedder": {"text": query}, "prompt": {"query": query}})
    

        st.subheader("Найденные документы:")
        st.markdown(answer['generator']['replies'][0])
        # for doc in answer['retriever']['documents'][:3]:
        #     st.markdown(f"**{doc.content}**  \n🔗 [Источник]({doc.meta['link']})")
