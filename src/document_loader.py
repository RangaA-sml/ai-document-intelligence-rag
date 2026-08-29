from langchain_community.document_loaders import PyPDFLoader
import tempfile


def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    return documents