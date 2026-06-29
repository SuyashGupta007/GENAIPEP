from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
data = TextLoader("D:/GenAiPEP/Document loaders/notes.txt")
splitter = RecursiveCharacterTextSplitter(separator="",chunk_size=20,chunk_overlap=1)
docs=data.load()
chunks = splitter.split_documents(docs)
#print(chunks)
for i in chunks:
    print(i.page_content)
    print()
