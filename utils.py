from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, memory, uploaded_file, question):
    try:
        # load model
        model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

        # process uploaded files
        file_content = uploaded_file.read()
        temp_file_path = "temp.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # text block
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", ".", "!", "?", ",", ""]
        )
        texts = text_splitter.split_documents(docs)

        # embed generation and vector storage
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(texts, embeddings_model)
        retriever = db.as_retriever()

        # build qa chains
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
        )

        # generate answer
        response = qa({"chat_history": memory, "question": question})
        answer = response["answer"]

        return {
            "answer": answer,
            "chat_history": response["chat_history"]
        }

    except Exception as e:
        return {"error": f"오류s: {str(e)}"}
