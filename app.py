from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''


loader = UnstructuredPDFLoader('./field-guide-to-data-science.pdf')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "test1"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, openai_api_key=OPENAI_API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rqa = ConversationalRetrievalChain.from_llm(llm, docsearch.as_retriever(), memory=memory)

def retrieve_answer(query, chat_history):
    # retrieval_result = rqa({"question": query, "chat_history": chat_history})
    memory.chat_memory.add_user_message(query)
    
    res = rqa({"question": query})
    retrieval_result = res["answer"]

    error_strings = ["The given context does not provide", "I'm sorry", "context provided does not",
                    "the given context does not provide", "The provided context does not contain",
                    "I don't know", "This information is not provided", "in the given context"]
    
    if any(error in retrieval_result for error in error_strings):    
        print(type(query))
        base_result = llm.generate(query)
        return base_result
        # return base_result.generations[0][0].text
    else:
        """ sources = res["sources"]
        source_info = f" (source: {sources[0]})" if sources else ""
        return retrieval_result + source_info """
        return retrieval_result


# Define a function to display the chat messages and the retrieved sources
""" def display_with_sources(message, matches):
    print("Assistant: " + message)
    if len(matches) > 0:
        print("Relevant sources:")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match['metadata']['title']}: {match['metadata']['url']}") """

# Chat loop
messages = []

print("Welcome to the chatbot. Enter 'quit' to exit the program.")
while True:
    user_message = input("You: ")

    if user_message.lower() == "quit":
        break

    answer = retrieve_answer(user_message, messages)
    """  source_documents = answer['source_documents']
    print(source_documents) """
    print("Assistant:", answer)
    memory.chat_memory.add_ai_message(answer)
    messages.append((user_message, answer))

    """ if len(memory.chat_memory) > 5:
        memory.chat_memory.pop(0) """