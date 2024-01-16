from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import streamlit as st

GOOGLE_API_KEY = st.secrets["API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

few_shots = {
    "What is the length of the song Man In The Box?": "SELECT (Milliseconds / 60000) AS minutes, (Milliseconds % 60000) / 1000 AS seconds FROM Track WHERE Name = 'Man In The Box'",
    "Give me 5 random tracks from the Pop Genre": "SELECT TrackId, Name FROM track WHERE genreid = (SELECT genreid FROM genre WHERE name = 'Pop') ORDER BY RANDOM() LIMIT 5",
    "top 3 most popular songs":"SELECT Name, COUNT(PlayListTrack.TrackId) as PlaylistCount From Track JOIN PlaylistTrack ON Track.TrackId = PlaylistTrack.TrackId GROUP BY Track.TrackId ORDER BY PlaylistCount DESC LIMIT 3",
    "What is the total number of tracks in the database?": "SELECT COUNT(*) FROM Track",
}

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question], "question": question})
    for question in few_shots.keys()
]

vector_db = FAISS.from_documents(few_shot_docs, embeddings)
vector_db.save_local("vector_db")