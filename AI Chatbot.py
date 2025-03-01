import openai

def chat_with_gpt(prompt):
    """Function to interact with OpenAI's GPT model"""
    openai.api_key = "your-api-key-here"  # Replace with your API key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("AI Chatbot: Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat_with_gpt(user_input)
        print("Bot:", response)
