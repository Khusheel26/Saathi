from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function to generate a response
def generate_response(input_text, conversation_history=None, max_length=150):
    # Combine the conversation history and the user's input
    if conversation_history:
        input_text = conversation_history + input_text

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response from the model with a specified max_length
    response_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=50256)

    # Decode the response and return it
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text

# Start a conversation
conversation_history = ""
while True:
    user_input = input("You: ")
    conversation_history = conversation_history + user_input + " "
    response = generate_response(user_input, conversation_history=conversation_history, max_length=150)
    print("Chatbot: " + response)