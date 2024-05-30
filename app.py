import streamlit as st

def main():
    st.set_page_config(page_title="helloyal")
    st.header("chat with pdf-gpt")
    st.text_input("ask a question")

    with st.sidebar:
        st.subheader("your doc")
        st.file_uploader("upload ur pdfs")
        st.button("process")




if __name__== '__main__':
    main()