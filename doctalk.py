import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer
from itertools import zip_longest
import pyttsx3
import speech_recognition as sr
import os
import time


openapi_key = st.secrets["OPENAI_API_KEY"]

def capture_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Oops! It seems like I missed that. Can you please repeat?")
        return ""
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service. Please check your internet connection.")
        return ""
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""


def text_to_speech(text):
    engine = pyttsx3.init()

    # Set the voice to Microsoft Zira
    zira_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty('voice', zira_id)

    # Set the rate (speed) of speech. Lower values will slow down the speech.
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)  # You can adjust the 50 to your preference.

    engine.say(text)
    engine.runAndWait()



def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".csv":
            text += get_csv_text(uploaded_file)
        elif file_extension == ".txt":
            text += get_txt_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    df = pd.read_csv(file)
    text = df.to_string()
    return text


def get_txt_text(file):
    return file.getvalue().decode("utf-8")


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openapi_key, model_name='gpt-3.5-turbo-16k', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handel_userinput(user_question):
    # Initialize the progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Processing your question...")

    # You can simulate progress by incrementing periodically (this is just an example)
    for i in range(4):
        # Increment progress bar
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.5)  # This is just to slow down the progress for demonstration

    # Now, process the actual question
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})

    # Update progress bar to 100% once done
    progress_bar.progress(100)
    progress_text.text("Done processing!")

    st.session_state.chat_history = response['chat_history']
    # You can add a small delay or directly clear the progress display if you want
    time.sleep(1)
    progress_bar.empty()
    progress_text.empty()

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            key_prefix = "user_" if i % 2 == 0 else "assistant_"
            message(messages.content, is_user=(i % 2 == 0), key=key_prefix + str(i))


def render_chat_history():
    """Function to render the chat history."""
    response_container = st.container()

    if st.session_state.chat_history:  # Check if chat_history is not None
        with response_container:
            for i, messages in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(messages.content, is_user=True, key=str(i))
                else:
                    message(messages.content, key=str(i))



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.title("📄🔊 DocTalker a VoiceBot 🔍🗣")
    st.markdown("### 🔍Dive into 📚PDF, 📒DOCX, 📊CSV & 📝TXT content📢")
    st.markdown("")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True)
        process = st.button("Process")
    
    if process:
        if not openapi_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")

        # Determine the document type for the message
        doc_type = os.path.splitext(uploaded_files[0].name)[1][1:].upper()  # Gets the file extension and converts it to uppercase
        default_message = f"You have uploaded a {doc_type} document. How can I assist you?"
        text_to_speech(default_message)  # Speak the default message
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("File chunks created...")
        # create vector stores
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore, openapi_key)
        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        render_chat_history()  # Render the chat history
        
        if st.button("Ask via Voice"):
            user_input = capture_audio()
            if user_input:  
                handel_userinput(user_input)
                response_text = st.session_state.chat_history[-1].content
                st.session_state.last_response = response_text  # Store the response text
                text_to_speech(response_text)

        # Check if the 'last_response' key exists in the session state 
        # and if there's any content to play
        if 'last_response' in st.session_state and st.session_state.last_response:
            if st.button("Repeat Answer"):
                text_to_speech(st.session_state.last_response)

if __name__ == '__main__':
    main()



# with repeat final