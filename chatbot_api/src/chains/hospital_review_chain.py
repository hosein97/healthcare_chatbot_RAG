import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint #  not working (why does it need text-generation lib at all?!)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
print('here')

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),model_name="sentence-transformers/all-MiniLM-l6-v2")
print('here2')

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)
print('here-21')
review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

repo_id = "HuggingFaceH4/zephyr-7b-beta"

print('here3')

llm = HuggingFaceEndpoint(repo_id=repo_id)
print('here4')

chat_model = ChatHuggingFace(llm=llm)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt