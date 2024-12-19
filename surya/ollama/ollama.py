from ollama import Client
from surya.ollama.conf import OLLAMA_HOST, MODEL
client = Client(host=OLLAMA_HOST)


def chat_with_model(content, model=MODEL):
    response = client.chat(model=model, messages=[
        {
            'role': 'user',
            'content': content,
        },
    ])
    return response['message']['content']
