from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama 

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)


# Langchain monitoring, tracing 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API server"
)

# ollama llama3.2
llm1 = Ollama(model = "llama3.2:1b")


# ollama gemma3
llm2 = Ollama(model = "gemma3:1b")

prompt1 = ChatPromptTemplate.from_template("Write me a topic about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|llm1,
    path="/essay",
)

add_routes(
    app,
    prompt2|llm2,
    path="/poem",
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
    