from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Flash")
model = ChatHuggingFace(llm=llm)
res = model.invoke(
    "give me some import question of tree,graph,array that can be asked in TCS nqt 2026"
)
print(res.content)
