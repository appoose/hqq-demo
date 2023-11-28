from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import (
    AutoTokenizer, 
    TextIteratorStreamer
)
from hqq.models.vit_timm import ViTHQQ
from hqq.models.llama_hf import LlamaHQQ

from threading import Thread
from pydantic import BaseModel, HttpUrl, FilePath
from typing import Tuple, List, Optional
from sys import stdout
import os
import logging
import gc 

import torch
import numpy as np
import open_clip


from huggingface_hub import login as hf_login


app = FastAPI()
# Load models on startup
@app.on_event("startup")
async def startup_event():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)    
    global model_visual, model_chat, tokenizer_chat, tokenizer_visual
    try:
        
        # model_visual = ViTHQQ.from_quantized("mobiuslabsgmbh/Llama-2-13b-chat-hf-4bit_g64-HQQ")        
        # model_visual = ViTHQQ.from_quantized("mobiuslabsgmbh/CLIP-ViT-H-14-laion2B-2bit_g16_s128-HQQ")
        # orig_model, _ , preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')

        # tokenizer_visual  = open_clip.get_tokenizer('ViT-H-14')
        # model_text = orig_model.encode_text

        # model_id  = 'mobiuslabsgmbh/Llama-2-70b-chat-hf-2bit_g16_s128-HQQ'
        model_id = "mobiuslabsgmbh/Llama-2-13b-chat-hf-4bit_g64-HQQ"
        #Load the tokenizer
        tokenizer_chat = AutoTokenizer.from_pretrained(model_id)
        tokenizer_chat.use_default_system_prompt = False

        #Load the model
        model_chat     = LlamaHQQ.from_quantized(model_id)
        model_chat = torch.compile(model_chat)


    except:
        logging.error("Issue loading model")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class AsyncIteratorWrapper:
    def __init__(self, iterable):
        self._iterable = iterable

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._iterable)
        except StopIteration:
            raise StopAsyncIteration
        return value


class ChatPromptInput(BaseModel):
    prompt: str
    other_info : Optional[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    messages: List[ChatMessage]


def print_flush(data):
    stdout.write("\r" + data)
    stdout.flush()


TEMPLATES = (
    lambda c: f'itap of a {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
)

@torch.no_grad()
def forward_image(img):
    x = preprocess(img).unsqueeze(0)
    f = model_visual(x.half().cuda())
    f /= torch.norm(f, p=2, dim=-1, keepdim=True)
    return f

@torch.no_grad()
def forward_text(text_batch_list, normalize=True):
    inputs  = tokenizer(text_batch_list)
    f       = model_text(inputs)
    if(normalize):
        f  /= torch.norm(f, p=2, dim=-1, keepdim=True)
    del inputs
    return f.half().to('cuda')

def forward_text_with_templates(text, templates=TEMPLATES, normalize=True):
    f = forward_text([t(text) for t in templates], normalize=False).mean(axis=0)
    if(normalize):
        f  /= torch.norm(f, p=2, dim=-1, keepdim=True)
    return f

def classifier_zero_shot_with_pil(img, classes):
    classifiers  = torch.cat([forward_text_with_templates(c).reshape([1, -1]) for c in classes], axis=0)
    img_features = forward_image(img)
    scores       = torch.matmul(img_features, classifiers.T)[0].detach().cpu().numpy()
    out          = classes[np.argmax(scores)]
    return out

#Adapted from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py
def process_conversation(chat_messages):    
    system_prompt = "You are a helpful assistant."
    # chat_history  = []
    # message       = prompt

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
        conversation.extend(chat_messages)

    # for user, assistant in chat_history:
    #     conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    # conversation.append({"role": "user", "content": message})

    return tokenizer_chat.apply_chat_template(conversation, return_tensors="pt").to('cuda')


@app.post("/chat/")
async def chat(chat_messages: Chat) -> dict:
    async def event_stream():
        streamer = TextIteratorStreamer(tokenizer_chat, 
                                        timeout=10.0, 
                                        skip_prompt=True, 
                                        skip_special_tokens=True)
        print(chat_messages)
        generate_kwargs = dict(
            {"input_ids": process_conversation(chat_messages.messages)},
            streamer=streamer,
            max_new_tokens=1000,
            do_sample=False,
            top_p=0.90,
            top_k=50,
            temperature= 0.6,
            num_beams=1,
            repetition_penalty=1.2,
        )
        thread = Thread(target=model_chat.generate, kwargs=generate_kwargs)
        thread.start()

        async for new_text in AsyncIteratorWrapper(streamer):
            foo = new_text
            print(foo)
            yield foo
        
        del streamer
        gc.collect()
        torch.cuda.empty_cache()
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

