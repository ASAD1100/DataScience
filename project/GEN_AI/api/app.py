from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain.schema.runnable import RunnableSequence
from langchain_community.llms import Ollama 
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'TRUE'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

app = FastAPI(
    title = 'Langchain Server',
    version  ='1.0',
    description = 'A simple API Server'
)
model1 = Ollama(model = 'llama2')
model2 = Ollama(model = 'gemma3:1b')
prompt1 = ChatPromptTemplate.from_template('Write me an essay about {topic} with 100 words')
prompt2 = ChatPromptTemplate.from_template('Write me a poem about {topic} in shakespeare style')

add_routes(
    app, 
    RunnableSequence(prompt1|model1),
    path = '/essay'    
)

add_routes(
    app,
    RunnableSequence(prompt2|model2),
    path = '/poem'
)

if __name__ == '__main__':
    uvicorn.run(app,host = 'localhost',port = 8000)