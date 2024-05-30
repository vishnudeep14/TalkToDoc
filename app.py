'''
import streamlit as st
from dotenv import load_dotenv,find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from htmltemp import *
def get_pdf_text(pdf):
    text=""
    for i in pdf:
        pdf_reader=PdfReader(i)
        for j in pdf_reader.pages:
            text+=j.extract_text()
    return text

def get_chunk(raw_text):
    text_split=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_split.split_text(raw_text)
    return chunks

def vector_db(txt_chunk):
    #embd=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embd=OpenAIEmbeddings()
    vec_db=FAISS.from_texts(texts=txt_chunk,embedding=embd)
    return  vec_db

def get_conv_chain(vec_store):
    llm-ChatOpenAI()
    #llm =HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    mem=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conv_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vec_store.as_retriever(),
        memory=mem

    )
    return  conv_chain
def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv(find_dotenv())

    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    st.set_page_config(page_title="helloyal")
    st.write(css,unsafe_allow_html=True)
    st.header("chat with pdf-gpt")
    st.text_input("ask a question")
    st.write(user_template.replace("{{MSG}}","hELLO ROBOT"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","haaloww"),unsafe_allow_html=True)



    with st.sidebar:
        st.subheader("your doc")
        pdf_doc=st.file_uploader("upload ur pdfs",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing"):
                raw_text=get_pdf_text(pdf_doc)

                chunk_text=get_chunk(raw_text)

                vec_store=vector_db(chunk_text)

                st.session_state.conversation=get_conv_chain(vec_store)















if __name__== '__main__':
    main()

'''


import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()