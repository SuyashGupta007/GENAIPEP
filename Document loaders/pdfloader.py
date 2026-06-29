from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("D:/GenAiPEP/Document loaders/genAI pdf.pdf")

docs = data.load()
print(docs[0].page_content)