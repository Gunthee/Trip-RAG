from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TYPHOON_API_KEY")

def generate_response(prompt):
    client = OpenAI(api_key=api_key,
                    base_url="https://api.opentyphoon.ai/v1",
                    )
   
    messages = [
        {"role": "system", "content": "You are a helpful assistant for answering questions about tours."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=512,
        top_p=0.6,
        frequency_penalty=0,
        stream=False
    )


    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = "What are the best tours in Thailand?"
    response = generate_response(prompt)
    print(response)