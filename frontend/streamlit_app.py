import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Document Assistant")
st.caption("Upload documents and ask questions about them")

# initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Upload section
st.header("Upload a Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_files:
    with st.spinner("Processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )

    if response.status_code == 200:
        data = response.json()
        st.session_state.uploaded_files.append(uploaded_file.name)
        st.success(f"Successfully uploaded: {data['filename']}")
        with st.expander("View document summary"):
            st.write(data["summary"])
    else:
        st.error("Failed to upload document")

if st.session_state.uploaded_files:
    st.info(f"Documents in library: {', '.join(st.session_state.uploaded_files)}")

# Query section
st.header("Ask a Question")
question = st.text_input("What would you like to know?")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            )
        if response.status_code == 200:
            data = response.json()
            st.write("### Answer")
            st.write(data["answer"])
        else:
            st.error("Failed to get an answer")
    else:
        st.warning("Please enter a question")