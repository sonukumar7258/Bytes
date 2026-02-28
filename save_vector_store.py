import json
from dotenv import load_dotenv
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from utilities.github_helper_functions import flatten_repo_data
from langchain_community.vectorstores import FAISS
from utilities.jira_helper_functions import flatten_corpus
from utilities.notion_helper_functions import parse_dict, remove_keys_from_dict, keys_to_remove
from utilities.website_helper_functions import custom_chunking_website
from langchain_openai.llms import OpenAI
from langchain_core.documents import Document
import langchain
FAISS.allow_dangerous_deserialization = True

load_dotenv()

def load_corpus(file_path: str) -> dict:
    """
    Load a JSON corpus from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def split_chunks(strings: list, chunk_size: int = 1500, overlap: int = 300) -> list:
    """
    Split a list of strings into chunks of a maximum size, with optional overlap.

    Args:
        strings (list): The list of strings to chunk.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1500.
        overlap (int, optional): The amount of overlap between chunks. Defaults to 300.

    Returns:
        list: The list of chunked strings.
    """
    updated_list = []
    for s in strings:
        while len(s) > chunk_size:
            updated_list.append(s[:chunk_size])
            s = s[chunk_size - overlap:]
        if s:
            updated_list.append(s)
    return updated_list

# Load corpora from JSON files
notion_corpus = load_corpus('corpus/notion_corpus.json')
notion_cleaned = remove_keys_from_dict(notion_corpus, keys_to_remove)
jira_corpus = load_corpus('corpus/jira_corpus.json')
github_corpus = load_corpus('corpus/github_corpus.json')

# Process corpora
notion_text = ['notion\n' + parse_dict(page) for page in notion_cleaned]
# website_text = custom_chunking_website('website_data')
jira_text = flatten_corpus(jira_corpus)
github_text = flatten_repo_data(github_corpus)
notion_text = split_chunks(notion_text, 1500, 300)

def save_embeddings(name: str) -> None:
    """
    Save embeddings for a list of documents to a file.

    Args:
        name (str): The name of the file to save the embeddings to.
    """
    list_of_documents = []
    for text in notion_text:
        list_of_documents.append(Document(page_content=text, metadata=dict(source="notion")))
    for text in jira_text:
        list_of_documents.append(Document(page_content=text, metadata=dict(source="jira")))
    # for text in website_text:
    #     list_of_documents.append(Document(page_content=text, metadata=dict(source="website")))
    for text in github_text:
        list_of_documents.append(Document(page_content=text, metadata=dict(source="github")))
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(list_of_documents, embeddings)
    vector_store.save_local(f"embeddings/{name}")

# Save embeddings to file
save_embeddings('all')