from langchain_core.prompts import ChatPromptTemplate # to create structured prompt for chatbot
from fastapi import FastAPI 
from langchain_core.output_parsers import StrOutputParser # to parse the output from chatbot
from langchain_groq import ChatGroq # for the chat module
import os                          # for operating system work
from dotenv import load_dotenv     # to load environment variables  
from langserve import add_routes
from pydantic import BaseModel # Import BaseModel for request body definition

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Create Model from Groq, using the API Key and Model name
model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
#  Initialize Parser
parser = StrOutputParser()

# 1. Create Prompt Template 
generic_template ="Translate Message into the {language}:"
prompt = ChatPromptTemplate(
    [("system",generic_template),
     ("user","{text}")]
)
# 2. Create Chain
chain = prompt|model|parser

#  App Defination
# This line creates an instance of your FastAPI application.
# app is now your web application.
# You're providing some metadata:
# title: The name that will appear in your API documentation.
# version: The version of your API.
# description: A brief explanation of what your API does.

app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API using Langchain runnable interfaces"
            )


# This is where langserve helps integrate your chain (which is a LangChain object that takes 
# a prompt, sends it to a model, and parses the output) directly into your FastAPI application.
# It essentially creates an API endpoint (or routes) for your chain.
# app: Your FastAPI application instance.
# chain: The LangChain object you want to expose as an API.
# path="/langchain-chain": This defines the URL path where your chain will be accessible. 
# So, if your app runs on http://127.0.0.1:8000, this chain will be available at
# http://127.0.0.1:8000/langchain-chain.
# playground_type="default": This enables a web-based "playground" or 
# UI for testing this specific LangChain endpoint

add_routes(
    app,
    chain,
    path="/langchain-chain", 
    playground_type="default" # Ensures playground is available
)

# Option B: Add a direct FastAPI route to expose the chain in /docs
# This is if you specifically want it to show up in the default /docs
class TranslateRequest(BaseModel):
    text: str
    language: str

@app.post("/translate/", summary="Translate text into a specified language")
async def translate_text(request: TranslateRequest):
    """
    Translates the given text into the specified language using the LangChain model.
    """
    result = await chain.ainvoke({"text": request.text, "language": request.language})
    return {"translated_text": result}
# ---------------------------------------------------------------
# This is the standard way to run a FastAPI application.
# uvicorn is a lightning-fast ASGI (Asynchronous Server Gateway Interface) server 
# that FastAPI uses to actually serve your web application.
# uvicorn.run(app, ...): This tells Uvicorn to run your app (the FastAPI instance).
# host="127.0.0.1": Your application will be accessible only from your local machine.
# port=8000: Your application will listen for requests on port 8000.
# When you run this Python script, Uvicorn starts the server, and your FastAPI application 
# becomes active and ready to receive requests.
if __name__=="__main__":
     import uvicorn
     uvicorn.run(app,host="127.0.0.1",port=8000) # this is how we run the FastAPI
