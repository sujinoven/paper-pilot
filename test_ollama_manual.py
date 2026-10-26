from ollama import Client

client = Client(host="http://localhost:11434")

response = client.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful AI research assistant. "
                "Explain technical concepts clearly and concisely."
            ),
        },
        {
            "role": "user",
            "content": (
                "What is positional encoding in the "
                "Transformer architecture? "
                "Explain in 3 sentences."
            ),
        },
    ],
    options={
        "temperature": 0.2,
        "num_predict": 150,
    },
)

print("\nMODEL RESPONSE")
print("=" * 60)
print(response.message.content)