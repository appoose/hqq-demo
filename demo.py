import streamlit as st
import requests
import json 
import GPUtil
import time
import plotly.express as px
import datetime
from streamlit_autorefresh import st_autorefresh
import pandas as pd


st.title("HQQ model demos")
base_url = "https://a55fd6b38dc5.ngrok.app"
chat_url  = f"{base_url}/chat/"
headers = {"Content-Type": "application/json"}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize GPU data list in session state
if 'gpu_data_history' not in st.session_state:
    st.session_state.gpu_data_history = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_gpu_info():
    gpus = GPUtil.getGPUs()
    return [{'GPU': gpu.id, 'Memory Used (MB)': gpu.memoryUsed} for gpu in gpus]

# Sidebar for real-time GPU usage
with st.sidebar:
    st.title("Real-time GPU Usage")
    gpu_chart = st.empty()
    refresh_rate = 2  # Refresh rate in seconds


# Accept user input
if prompt := st.chat_input("LLama-13-B 4-Bit Quantized model: AMA ( eg: Tell me a Dad joke ) "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat_response = requests.post(chat_url,
                                headers=headers,
                                data=json.dumps({"prompt":prompt}),
                                stream=True)


        for line in chat_response:            
            if line: 
                full_response += (line.decode('utf-8') or "")
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})



# GPU info update
refresh_rate = 2  # Refresh rate in seconds
if st_autorefresh(interval=refresh_rate * 1000, key="gpu_info_refresh"):
    current_gpu_data = get_gpu_info()
    timestamp = datetime.datetime.now()  # Get current time

    # Append current data with timestamp to the history
    for gpu in current_gpu_data:
        st.session_state.gpu_data_history.append({
            'Time': timestamp,
            'GPU': gpu['GPU'],
            'Memory Used (MB)': gpu['Memory Used (MB)']
        })

    # Convert the data history to a DataFrame
    df_gpu_data = pd.DataFrame(st.session_state.gpu_data_history)

    # Plot the evolving time series
    fig = px.line(df_gpu_data, x='Time', y='Memory Used (MB)', color='GPU',
                  labels={'Memory Used (MB)': 'Memory Usage (MB)'},
                  title='GPU Memory Usage Over Time')
    gpu_chart.plotly_chart(fig, use_container_width=True)