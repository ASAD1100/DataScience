

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper)

wiki.name

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents(docs)
db = FAISS.from_documents(documents, OllamaEmbeddings(model= 'bge-m3'))
retriever = db.as_retriever()
retriever

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, 'langsmith_search',
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

retriever_tool.name

# Arxiv tool 
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun 

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

tools = [wiki, arxiv, retriever_tool]

from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model = 'llama3.1:8b')

from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Agents
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Agent executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)


agent_executor.invoke({'input':'Tell me about langsmith'})

print('\n\n')

agent_executor.invoke({'input':'What is Transformer model'})