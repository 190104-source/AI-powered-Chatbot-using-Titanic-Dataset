import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()
st.set_page_config(page_title="Titanic AI Analyst", layout="wide")


GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()


stats_context = f"""
Titanic Dataset Summary:
- Total Passengers: {len(df)}
- Gender: {len(df[df['Sex']=='male'])} males, {len(df[df['Sex']=='female'])} females
- Survival Rate: {(df['Survived'].mean()*100):.1f}%
- 1st Class: {len(df[df['Pclass']==1])}, 2nd: {len(df[df['Pclass']==2])}, 3rd: {len(df[df['Pclass']==3])}
- Average Fare: ${df['Fare'].mean():.2f}
- Ports: S=Southampton, C=Cherbourg, Q=Queenstown
"""


if GROQ_API_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )
else:
    st.error(" GROQ_API_KEY not found. Please add it to your .env file or Streamlit Secrets.")


st.title("Titanic Chatbot")
st.markdown("---")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask about survival rates, age, or ports..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
       
        full_query = f"Context: {stats_context}\nQuestion: {prompt}\nRule: Answer in one short sentence."
        try:
            ai_response = llm.invoke(full_query)
            answer = ai_response.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
           
            p = prompt.lower()
            if "age" in p:
                st.subheader("Age Distribution")
                age_bins = df['Age'].dropna().value_counts(bins=10).sort_index()
                st.bar_chart(pd.DataFrame({"Count": age_bins.values}, index=[str(i) for i in age_bins.index]))
            
            elif "survival" in p or "survive" in p:
                st.subheader("Survival Count by Ticket Class")
                st.bar_chart(df.groupby('Pclass')['Survived'].sum(), color="#2ecc71")
                
            elif "port" in p or "embark" in p:
                st.subheader("Passengers per Port")
                st.bar_chart(df['Embarked'].value_counts(), color="#3498db")
                
        except Exception as e:
            st.error(f"Error: {e}")