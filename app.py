import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
from langchain_community.vectorstores import FAISS
# Assuming 'file_path' is the path to your pickle file


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Result: ", response["output_text"])




def main():
    ## css
    st.markdown(
        """
        <style>
        .stApp {{background-image: linear-gradient(to top, #fbc2eb 0%, #a6c1ee 100%);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
            
          
        </style>
        """,
        unsafe_allow_html=True
    )

custom_css = """
<style>
body {
    font-family: Georgia, serif;
    font-size: 20px;
    background-color: #57ffdb;
}
header {
    background-color: #2c3e50;
    color: #ffffff;
    padding: 10px;
    border-bottom: 2px solid #1abc9c;
}
.stButton button {
    background-color: #ffe600;
    color: blue;
    border: none;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 25px;
    margin: 4px 2px;
    font-weight: bold;
    transition-duration: 0.4s;
    cursor: pointer;
}
.stButton button:hover {
    background-color: #b0fcf2;
    color: black;
    border: 4px solid #4CAF50;
}
.bold-text {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 30   px;  /* Font size */
    font-weight: bold;  /* Bold text */
    color: #333;  /* Text color */
}

</style>
"""

# Inject CSS with markdown

st.markdown(custom_css, unsafe_allow_html=True)
css = """
<style>
/* Hide the default sidebar toggle */
header .css-1q1n0ol { 
    display: none;
}

/* Custom sliding sidebar */
#sidebar {
    position: fixed;
    left: -250px;
    top: 0;
    bottom: 0;
    width: 250px;
    background-color: #2c3e50;
    color: ##2c3e50;
    transition: left 0.3s ease;
    padding: 20px;
    z-index: 1000;
}

#sidebar.active {
    left: 0;
}

#sidebar h1, #sidebar h2, #sidebar h3, #sidebar h4, #sidebar h5, #sidebar h6 {
    color: ##2c3e50;
}

#sidebar a {
    color: #1abc9c;
    text-decoration: none;
}

#sidebar a:hover {
    color: #16a085;
    text-decoration: underline;
}

#sidebar-toggle {
    position: absolute;
    left: 260px;
    top: 20px;
    background-color: #2c3e50;
    color: #ffffff;
    border: none;
    padding: 10px;
    cursor: pointer;
    border-radius: 5px;
}
</style>
"""

# JavaScript for sliding sidebar
js = """
<script>
function toggleSidebar() {
    var sidebar = document.getElementById('sidebar');
    if (sidebar.classList.contains('active')) {
        sidebar.classList.remove('active');
    } else {
        sidebar.classList.add('active');
    }
}
</script>
"""

# Inject CSS and JavaScript
st.markdown(css, unsafe_allow_html=True)
st.markdown(js, unsafe_allow_html=True)
   
st.markdown(
"""
    <style>
    
.main {
    background-image: linear-gradient(to top, #a18cd1 0%, #fbc2eb 100%);
    font-family: Georgia, serif;
}

h1 {
    color: #a30000a1;
    font-family: Georgia, serif;
    text-align: left;
    font-size: 30px;
}

.stButton>button {
    background-color: #b0fcf2;
    color: black;
    font-size: 30px;
    animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }
.stTextInput>div>div>input {
    font-family: Georgia, serif;
    font-size: 25px;

</style>
""",

    unsafe_allow_html=True
)

st.title("Chatbot with Documents📄")
st.write("Upload a file and ask the bot questions.")
user_question = st.text_input("Enter the Question ")
if user_question:
        user_input(user_question)

with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit ", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()