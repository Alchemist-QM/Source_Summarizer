import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory

# LLM setup
openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=openai_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load transcript and summary
def load_json_transcipt_summary():
    with open("final_output.json", "r") as f:
        data = json.load(f)
        timeline = data["timeline"]
        summary = data["final_summary"]
        video_info = data["video_info"]
    return timeline, summary, video_info


def chunk_transcript(timeline, embedder):
# Chunk transcript
    chunks = []
    metadatas = []
    for entry in timeline:
        for t, ts in zip(entry["transcript"], entry["timestamp"]):
            chunks.append(t)
            metadatas.append({"frame": entry["frame"], "frame_time": entry["frame_time"], "timestamp": ts})

    # Embedding model
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, metadatas, index

def retrieve(query,
            index,
            chunks, 
            metadatas, 
            embedder, 
            top_k=5):
    query_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]], [metadatas[i] for i in I[0]]

def run_youtube_chatbot():
    timeline, summary, video_info = load_json_transcipt_summary()
    chunks, metadatas, index = chunk_transcript(timeline, embedder)
    

    # New memory usage (LangChain >= 0.1.0)
    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=chat_history)

    print(f"Chatting with video: {video_info.get('title', 'Unknown')}")
    print(f"Summary: {summary}\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        retrieved_chunks, meta = retrieve(
            user_input, 
                                        index,
                                        chunks, 
                                        metadatas, 
                                        embedder
                                        )
        context = "\n".join(retrieved_chunks)
        prompt = f"""
    You are a helpful agent answering questions about a YouTube video.
    Video summary: {summary}
    Relevant transcript: {context}
    User question: {user_input}
    """
        response = llm.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        print(f"Agent: {response.choices[0].message.content}\n")
        # Optionally store chat messages in memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.choices[0].message.content)
        
        
