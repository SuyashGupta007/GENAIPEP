from langchain_community.document_loaders import TextLoader


loader = TextLoader("D:/GenAiPEP/Document loaders/notes.txt")

docs= loader.load()
print(len(docs))
#print(docs[0].page_content)