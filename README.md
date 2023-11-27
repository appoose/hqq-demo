# Simple Demo for HQQ Models
## Overview
This repository showcases a basic implementation of models quantized using the [HQQ (Half Quadratic Quantization)](https://github.com/mobiusml/hqq) method. 

## Components
The repository includes two main components:

FastAPI Backend: A lightweight and efficient API framework is used to serve the quantized models. This FastAPI implementation ensures easy deployment and scaling.
Streamlit Frontend: An interactive frontend built with Streamlit, providing a user-friendly interface to interact with the quantized models.
Purpose
This demo serves as a practical accompaniment to the HQQ Blog, which offers detailed insights and tutorials on model quantization using HQQ. The aim is to provide a hands-on example for users to understand how HQQ can be integrated and utilized in real-world applications.

## Getting Started
### Prerequisites
Python 3.6 or later
Knowledge of FastAPI and Streamlit (optional but beneficial)

### Installation
Clone this repository:
```bash
git clone https://github.com/yourusername/simple-demo-for-hqq-models.git
```
Install required packages:
```bash 
pip install -r requirements.txt
```

### Environment Setup

Before running the FastAPI server, you need to set an environment variable for your Hugging Face token. This token is necessary for authentication with the Hugging Face model repository.

Set the HF_TOKEN environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

### Running the Application
Start the FastAPI server:

```bash
uvicorn main:app --reload
```

### Launch the Streamlit frontend:

```bash
streamlit run app.py
````

### Usage
Once both the FastAPI server and Streamlit frontend are running, navigate to the provided local URL to interact with the quantized models. You can input data, receive predictions, and visualize the performance of the HQQ models.

### Contributions
Contributions to this repository are welcome. Please refer to the CONTRIBUTING.md file for guidelines.

### License
This project is licensed under the MIT License.

### Acknowledgments
Special thanks to the [@mobicham](https://github.com/mobicham) the main contributorof the HQQ project for their pioneering work in model quantization.


