

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''

loader = UnstructuredPDFLoader('./field-guide-to-data-science.pdf')
data = loader.load()
# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
texts = text_splitter.split_documents(data)

# print (f'Now you have {len(texts)} documents')


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "chatconnect1"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7, openai_api_key=OPENAI_API_KEY)
# chain = load_qa_chain(llm, chain_type="stuff")

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain



""" rqa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
) """
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rqa = ConversationalRetrievalChain.from_llm(llm, docsearch.as_retriever(), memory=memory)



# Define a function to retrieve an answer to a question
""" def retrieve_answer(query):
    retrieval_result = rqa.run(query=query) """
def retrieve_answer(query, chat_history):
    # retrieval_result = rqa({"question": query, "chat_history": chat_history})
    
    res = rqa({"question": query, "chat_history": chat_history})
    retrieval_result = res["answer"]

    # Check if the answer is satisfactory
    if "The given context does not provide" in retrieval_result or "context provided does not" in retrieval_result or "the given context does not provide" in retrieval_result or "The provided context does not contain" in retrieval_result or "I don't know" in retrieval_result or "This information is not provided" in retrieval_result or "in the given context" in retrieval_result:
        # If not, use the base GPT-3.5 Turbo model to answer the question
        base_result = llm.generate([query])
        # return base_result['generations'][0][0]['text']
        return base_result
    else:
        # return retrieval_result
        return retrieval_result


# Define a function to display the chat messages and the retrieved sources
""" def display_with_sources(message, matches):
    print("Assistant: " + message)
    if len(matches) > 0:
        print("Relevant sources:")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match['metadata']['title']}: {match['metadata']['url']}") """


# Start the chat loop
messages = []

print("Welcome to the chatbot. Enter 'quit' to exit the program.")
while True:
    user_message = input("You: ")

    if user_message.lower() == "quit":
        break

    # Retrieve the answer to the user's question
    answer = retrieve_answer(user_message, messages)
    print("Assistant:", answer)

    messages.append((user_message, answer))

    # Add the user's message and the chatbot's answer to the messages
