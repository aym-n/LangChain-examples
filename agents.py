from langchain_google_genai import GoogleGenerativeAI

GOOGLE_API_KEY="AIzaSyBK5BgMfYwlZQNQBFcHFNzGRNRaD1M9598"

from langchain.agents import create_sql_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from langchain_core.messages import AIMessage, HumanMessage

db = SQLDatabase.from_uri("sqlite:///chinook.db")
suffix = """Make sure to provide all the responses in markdown format
current conversation:
{chat_history}
Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

agent_executor = create_sql_agent(
    llm=GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/text-bison-001"),
    toolkit=SQLDatabaseToolkit(db = db, llm=GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="models/text-bison-001")),
    verbose=True,
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    suffix=suffix,
)

import streamlit as st
chat_history = []
st.title("Chinook Database Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        message_placeholder = st.empty()
        message = agent_executor.invoke({"input": prompt, "chat_history": chat_history})
        message_placeholder.markdown(message["output"])
    st.session_state.messages.append({"role": "assistant", "content": message['output']})
    chat_history.append(AIMessage(content=message['output']))  



