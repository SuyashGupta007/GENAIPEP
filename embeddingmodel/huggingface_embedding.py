from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = ["Hello i am Suyash Gupta", "Hi i am teaching GenAI", "All are great students"]

vector = embeddings.embed_documents(texts)
# print(vector)
# print(len(vector))
print(vector[0])
