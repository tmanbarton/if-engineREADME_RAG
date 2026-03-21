import json
import os
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

system_message = """
        Your only task is to answer questions about the if-engine Java library, a reusable Java-based engine and builder for creating interactive fiction games, via its README.
        Make sure you identify examples from actual documentation so you don't try to tell the user misleading information that will restrict them in their usage of the library.
        Example: there's an example with a lockable door and the user may about lockable things. You don't say there are lockable doors. You give information about lockable objects in general, otherwise you may lead the user to think the only objects they can lock are doors.
        Also note, this is CLI based, so you don't have access to em dashes. Use "word - word" for em dashes instead of "word-word".
        """

# Get if-engine README embeddings from local json file as an array
with open("index.json") as f:
    index = json.load(f)

chunks = [item["text"] for item in index]
embeddings = np.array([item["embedding"] for item in index])
model = SentenceTransformer("all-MiniLM-L6-v2")

# The chatbot. Loop until "quit" is entered or program is stopped
# Add an initial message to the message history and print it to the console to prompt the user
initial_message = "How can I help you with the if-engine Java library?"
messages = [{"role": "assistant", "content": initial_message}]
while True:
    print(initial_message)
    user_input = input("~ ")
    if user_input == "quit" or user_input == "exit":
        break

    # Embed user input and compare to stored embedding to get closest 3 matches
    input_embedding = model.encode([user_input])[0]
    scores = np.dot(embeddings, input_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding))
    top_3 = np.argsort(scores)[::-1][:3]

    # Prepare user message to send to LLM. Contains user input and the documentation found
    context = "\n\n".join(chunks[i] for i in top_3)
    augmented_input = f"Question:\n{user_input}\n\nDocumentation context:\n{context}"
    messages.append({"role": "user", "content": augmented_input})

    # Send message to LLM
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model = "claude-haiku-4-5",
        max_tokens = 1024,
        system = system_message,
        messages = messages)

    # Replace user message (last element in message array) with only user input. Don't want documentation in conversation history.
    # Append assistant message to conversation history and print
    messages[-1]["content"] = user_input
    response_text = response.content[0].text
    messages.append({"role": "assistant", "content": response_text})
    print(f"~ {response_text}")



