from databse import *
from decouple import config
import json
import requests
from openai import OpenAI
 
def process_and_respond_to_chat(user_id, session_id,phone_number):
    # Initialize configurations
    TOKEN = config("TOKEN")
    OPENAI_API_KEY = config("OPENAI_API_KEY")
    WHATSAPP_API_URL = config("WHATSAPP_URL")
   
    # Retrieve chat history from the database
    chat_history = retrieve_chat_history(user_id, session_id)
   
    # Format chat history into the desired structure for OpenAI input
    formatted_string = "\n".join([f"{entry[0]} : {entry[1]}" for entry in chat_history])
 
    # Analyze user intent using OpenAI's API
    client = OpenAI(api_key=OPENAI_API_KEY)
 
    print("Analyzing the user intent...")
    prompt = [
    {"role": "system", "content": "You are Placecom Chatbot, a friendly, helpful, and polite AI assistant. Your task is to generate a one-line message that keeps the conversation flowing naturally, based on the chat history. Your responses should be context-aware and upbeat, including relevant and engaging emojis that match the tone and content of the conversation. Be sure to offer answers that are short, clear, and interactive! 😊,please provide the result in less then one line or -34 words"},
    {"role": "user", "content": f"Here is the chat history:\n{formatted_string}\n\nGenerate a single, friendly, and context-aware WhatsApp response in one line with emojis based on the conversation above."}
]
 
    response = client.chat.completions.create(model="gpt-3.5-turbo-16k", messages=prompt, temperature=0.3)
    result = response.choices[0].message.content.strip().lower()
 
    # Print the chat history and response for debugging purposes
    print("####################################", chat_history)
    print("Generated Response:", result)
 
 
process_and_respond_to_chat("user_918847444813", "3aa789fb-f5ee-4010-a113-dc21875ad17e")