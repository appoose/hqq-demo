import streamlit as st
import requests
import json 
import GPUtil
import time
import plotly.express as px
import datetime
from streamlit_autorefresh import st_autorefresh
import pandas as pd


# st.title("HQQ model demos")
base_url = "https://a55fd6b38dc5.ngrok.app"
chat_url  = f"{base_url}/chat/"
headers = {"Content-Type": "application/json"}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# @st.cache(allow_output_mutation=True)
# @st.cache_data
# def get_gpu_data_history():
#     return []

# Initialize or get the cached GPU data history
# gpu_data_history = get_gpu_data_history()
# Function to fetch and append current GPU info

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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



# # GPU info update
# refresh_rate = 2  # Refresh rate in seconds
# if st_autorefresh(interval=refresh_rate * 1000, key="gpu_info_refresh"):
#     append_current_gpu_info()

#     df_gpu_data = pd.DataFrame(gpu_data_history)

#     # Plot the evolving time series
#     fig = px.line(df_gpu_data, x='Time', y='Memory Used (MB)', color='GPU',
#                   labels={'Memory Used (MB)': 'Memory Usage (MB)'},
#                   title='GPU Memory Usage Over Time')
#     gpu_chart.plotly_chart(fig, use_container_width=True)