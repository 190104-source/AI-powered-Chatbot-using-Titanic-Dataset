import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
st.set_page_config(page_title="Titanic AI Chatbot", layout="wide")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df = load_data()


def handle_visuals(user_prompt):
    prompt = user_prompt.lower()
    
    if "age" in prompt or "histogram" in prompt:
        with st.expander("📊 View Age Distribution", expanded=True):
            
            age_counts = df['Age'].dropna().value_counts(bins=10).sort_index()
            chart_df = pd.DataFrame({"Count": age_counts.values}, index=[str(c) for c in age_counts.index])
            st.bar_chart(chart_df)
            st.caption("Distribution of passenger ages divided into 10 bins.")

    elif "survival" in prompt or "survive" in prompt:
        with st.expander("View Survival by Class", expanded=True):
            survival_data = df.groupby('Pclass')['Survived'].sum()
            st.bar_chart(survival_data, color="#2ecc71")
            st.caption("Total number of survivors for each ticket class.")

    elif "port" in prompt or "embark" in prompt:
        with st.expander("View Passengers per Port", expanded=True):
            port_counts = df['Embarked'].value_counts()
            st.bar_chart(port_counts, color="#3498db")
            st.caption("Count of passengers who embarked from each location.")


st.title("Titanic Chatbot")

if prompt := st.chat_input("Ask about the Titanic..."):
    st.chat_message("user").markdown(prompt)
    
    with st.chat_message("assistant"):
        
        if GROQ_API_KEY:
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            
            
            context = f"Total Passengers: {len(df)}, Survival Rate: {(df['Survived'].mean()*100):.1f}%"
            response = llm.invoke(f"Context: {context}\nQuestion: {prompt}\nRule: Answer in one short sentence.")
            
            st.markdown(f"**AI Analyst:** {response.content}")
            handle_visuals(prompt) 
        else:
            st.error("API Key missing!")