# Rename `os.environ` to `env` for nicer code
from os import environ as env
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv,find_dotenv
import markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from util import get_serper_api_key

from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv('config/.env')



# Rename `os.environ` to `env` for nicer code

print('GOOGLE_API_KEY:  {}'.format(env['GOOGLE_API_KEY']))

GEMINI_API_KEY = env['GOOGLE_API_KEY']

env["GEMINI_API_KEY"] = env['GOOGLE_API_KEY']
MODEL_NAME = "models/text-embedding-004"

env["SERPER_API_KEY"] = env['SERPER_KEY']


llm = LLM(
    model = "gemini/gemini-1.5-pro-latest",
    temperature = 0.7
)