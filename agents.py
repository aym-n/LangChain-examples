from langchain_google_genai import GoogleGenerativeAI

GOOGLE_API_KEY="AIzaSyBK5BgMfYwlZQNQBFcHFNzGRNRaD1M9598"

from langchain.agents import create_sql_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///chinook.db")

agent_executor = create_sql_agent(
    llm=GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/text-bison-001"),
    toolkit=SQLDatabaseToolkit(db = db, llm=GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/text-bison-001")),
    verbose=True,
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.invoke("name the top 5 albums with the most tracks and the number of tracks in each album")