import streamlit as st

st.title("SQLang 🔮")
st.write("Use the chatbot to ask questions about the Chinook database")
st.caption ("You can ask Questions like 'What is the total number of tracks in the database?' or 'List all the tables in the database'")

st.sidebar.title("Settings")

model = st.sidebar.selectbox(
    'LLM',
    ('models/text-bison-001', 'gemini-pro'))

method = st.sidebar.selectbox(
    'LangChain Methods',
    ('SQL Chain + FewShot', 'SQL Agent'))

k = st.sidebar.slider('Number of examples to select for FewShot', 1, 5, 2, 1)

st.divider()

st.sidebar.title("About")

st.sidebar.info(" SQLang 🔮 allows user to write seamless natural language database queries,  powered by LangChain, Streamlit, and Google's LLM models (Gemini-Pro and Text-Bison).  it uses LangChain's  zero-shot SQL agent and a SQL database chain with is enhanced using few-shot examples.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How can i help you?"})


from langchain_google_genai import GoogleGenerativeAI

GOOGLE_API_KEY = st.secrets["API_KEY"]

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

llm= GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=model)

from langchain_community.utilities.sql_database import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///chinook.db")


from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
vector_db = FAISS.load_local("vector_db",  embeddings)
selector = SemanticSimilarityExampleSelector(vectorstore=vector_db, k=k)

from langchain.chains.sql_database.prompt import _sqlite_prompt, PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate

prompt = PromptTemplate(
    input_variables=["sql_query", "question"],
    template="Question: {question}\nSQL Query: {sql_query}\n",
)

from langchain.prompts import FewShotPromptTemplate
fewShot = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=prompt,
    suffix=PROMPT_SUFFIX,
    prefix=_sqlite_prompt + """Return the result of the SQL query in Markdown format.""",
    input_variables=["input", "table_info", "top_k"],
)


from langchain_experimental.sql import SQLDatabaseChain
fewShotChain = SQLDatabaseChain.from_llm(llm, db, prompt=fewShot, verbose=True)

from langchain.agents import create_sql_agent
SQLAgent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        message_placeholder = st.empty()
        if method == 'SQL Agent':
            message = SQLAgent.invoke(prompt)
            message_placeholder.markdown(message['output'])
            st.session_state.messages.append({"role": "assistant", "content": message["output"]}) 
        else:
            message = fewShotChain.invoke(prompt)
            message_placeholder.markdown(message['result'])
            st.session_state.messages.append({"role": "assistant", "content": message["result"]})


