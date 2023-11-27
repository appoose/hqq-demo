import streamlit as st
import requests
import json 
import GPUtil
import time

st.title("HQQ model demos")
base_url = "https://a55fd6b38dc5.ngrok.app"
chat_url  = f"{base_url}/chat/"
headers = {"Content-Type": "application/json"}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_gpu_info():
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        gpu_info = {
            'id': gpu.id,
            'name': gpu.name,
            'load': f"{gpu.load*100}%",
            'free memory': f"{gpu.memoryFree}MB",
            'used memory': f"{gpu.memoryUsed}MB",
            'total memory': f"{gpu.memoryTotal}MB",
            'temperature': f"{gpu.temperature} °C"
        }
        list_gpus.append(gpu_info)
    return list_gpus

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

if st.button('Refresh GPU Info'):
    gpu_info = get_gpu_info()
    for gpu in gpu_info:
        st.write(f"GPU {gpu['id']}: {gpu['name']}")
        st.write(f"Load: {gpu['load']}")
        st.write(f"Memory Usage: {gpu['used memory']} / {gpu['total memory']}")
        st.write(f"Temperature: {gpu['temperature']}")
        


