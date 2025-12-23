
import os
import torch

#from langchain_openai import OpenAIEmbeddings
#from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import traceback

from dotenv import load_dotenv

###load environment variables
load_dotenv()

# API Key Configuration 
groq_api = os.getenv('GROQ_API_KEY')
tavily_api = os.getenv('TAVILY_API_KEY')
os.environ['TAVILY_API_KEY'] = tavily_api
mistralai_api = os.getenv('MISTRAL_API_KEY')
os.environ["MISTRAL_API_KEY"] = mistralai_api
hf_api = os.getenv('HF_TOKEN')
#print(mistralai_api)

torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', '_classes.py')]

# details here: https://openai.com/blog/new-embedding-models-and-api-updates
#openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
#embeddings = MistralAIEmbeddings(model="mistral-embed")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

loader = PyPDFLoader(file_path=r"C:\Tasks\Multimodal_Agentic_RAG\Agentic_CorrectiveRAG_LangGraph\Inputs\Heart_Health_Fact_Sheet.pdf")
docs = loader.load()

# Chunk docs
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = splitter.split_documents(docs)

chroma_db = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, persist_directory="./chromapdf_db")

similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 2, "score_threshold": 0.3})

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM for grading
#llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    #model="llama3-70b-8192",
    temperature=0
)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not and nothing else.
             """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}

                     User question:
                     {question}
                  """),
    ]
)

# Build grader chain
doc_grader = (grade_prompt
                  |
              structured_llm_grader)

# Create RAG prompt for response generation
prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer and don't give any explanation.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.

            Question:
            {question}

            Context:
            {context}

            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)

# Initialize connection with GPT-4o
#chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)
chatgpt = ChatGroq(
    model="llama-3.3-70b-versatile",
    #model="llama3-70b-8192",
    temperature=0
)
# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context')
                        |
                    RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
      |
    prompt_template
      |
    chatgpt
      |
    StrOutputParser()
)

#llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    #model="llama3-70b-8192",
    temperature=0
)

# Prompt template for rewriting
SYS_PROMPT = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
                 - strictly rewrite the improved question only, don't give any explanation or multiple rephrased questions, only give a single line best selected rephrased question.
             """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
                     {question}

                     Formulate an improved question.
                  """,
        ),
    ]
)
# Create rephraser chain
question_rewriter = (re_write_prompt
                        |
                       llm
                        |
                     StrOutputParser())

#tv_search = TavilySearchResults(max_results=3)#, search_depth='advanced',max_tokens=10000)
tv_search = DuckDuckGoSearchRun()

#class GraphState(TypedDict):
class GraphState(MessagesState):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """

    #messages: list[MessagesState]
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]
    remaining_steps:int

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    #print("retrieve state = = == ", state)
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]

    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.

    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    #print("grade_doc state = = == ", state)
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search_needed = "Yes"
                continue
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}

def rewrite_query(state):
    """
    Rewrite the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """
    #print("rewrite_query state = = == ", state)
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

from langchain.schema import Document

def web_search(state):
    """
    Web search based on the re-written question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    #print("web search state = = == ", state)
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = tv_search.invoke(question)
    web_results=docs
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

def generate_answer(state):
    """
    Generate answer from context document using LLM

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #print("gen_ans state = = == ", state)
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    #print("dec_ans state = = == ", state)
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]

    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"


agentic_rag = StateGraph(GraphState)

# Define the nodes
agentic_rag.add_node("retrieve", retrieve)  # retrieve
agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
agentic_rag.add_node("web_search", web_search)  # web search
agentic_rag.add_node("generate_answer", generate_answer)  # generate answer

# Build graph
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("generate_answer", END)

# Compile
agentic_crag = agentic_rag.compile(name="Agentic_CRAG")


llm_img = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

image_agent = create_react_agent(
    model=llm_img,
    tools=[],
    name="image_expert",
    prompt="Read the image and give insights or summary about the image from path C:\\Tasks\\Multimodal_Agentic_RAG\\Agentic_CorrectiveRAG_LangGraph\\Inputs\\heart_xray.jpg"
)

# Create supervisor workflow
workflow = create_supervisor(
    [agentic_crag, image_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a corrective rag and a image expert. "
        "For searching from web or pdf vectorstore as corrective rag based on user question, use Agentic_CRAG agentic_crag. "
        "For image analysis or anything with respect to image, use image_expert image_agent."
    ),
    #output_mode="full_history",
    state_schema=GraphState
)

# Compile and run
app = workflow.compile()

def run_rag(prompt):
    try:
        print("In Try...")
        result = app.invoke({
        "question":f"{prompt}",
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
        })

        img_keywords = ['Image Summary', 'Image Insights', 'Image Uploaded', 'Image Attached']

        if any(imgkey.lower() in prompt.lower() for imgkey in img_keywords):
            try:
                print("Keywords Found = = ", [imgkey for imgkey in img_keywords if imgkey.lower() in prompt.lower()])
                return result["messages"][-4].content
            except Exception as e:
                print("Exception Image = = = == = ", str(traceback.format_exc()))
                return result["messages"][-1].content
        
        return result["messages"][-1].content
    except Exception as e1:
        print("In Main Except...........")
        return str(traceback.format_exc())
    

import streamlit as st

st.title("Medical Agentic Corrective-RAG")
st.markdown(
    "Ask questions about Healthy Heart, Heart Attack, Healthy Diet, Image Analysis, Any Other Medical Topic, etc., The system uses a multi-agent approach to retrieve and verify information."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question about Healthy Heart, Heart Attack, Healthy Diet, Image Analysis, Any Other Medical Topic, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = run_rag(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})