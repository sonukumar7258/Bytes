from notion_client import Client
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Load Notion API key from environment variables
NOTION_API_KEY = os.getenv("NOTION_API_KEY")

# Initialize Notion client
notion = Client(auth=NOTION_API_KEY)

def fetch_block_children(block_id, notion):
    """
    Fetch all children blocks of a given block and include all block data recursively.

    Args:
        block_id (str): ID of the block to fetch children for.
        notion (notion_client.client.Client): Notion client instance.

    Returns:
        list: List of dictionaries containing data for each block.

    """
    block_children = notion.blocks.children.list(block_id=block_id)["results"]
    content = []
    for block in block_children:
        content.append(block)
        if "has_children" in block and block["has_children"]:
            block["children"] = fetch_block_children(block["id"], notion)
    return content

def fetch_all_pages(notion):
    """
    Fetch all standalone pages in the workspace and include all page and block data along with the URLs.

    Args:
        notion (notion_client.client.Client): Notion client instance.

    Returns:
        list: List of dictionaries containing data for each page.

    """
    pages = []
    query_results = notion.search(filter={"value": "page", "property": "object"})["results"]
    print("Fetching page details...")
    print(query_results)
    for page in query_results:
        page_id = page["id"]
        print(page_id)
        print(f"Processing page ID: {page_id}")
        page_details = {"page_data": page}
        page_url = page.get('url', 'URL not available')
        page_details["content"] = fetch_block_children(page_id, notion)
        page_details["url"] = page_url
        pages.append(page_details)

    return pages

def create_corpus(notion):
    """
    Create a corpus from all standalone pages in the workspace, including all data.

    Args:
        notion (notion_client.client.Client): Notion client instance.

    Returns:
        list: List of dictionaries containing data for each page.

    """
    corpus = fetch_all_pages(notion)
    return corpus

# Create Notion corpus
corpus = create_corpus(notion)

# Save the corpus to a JSON file
corpus_directory = os.path.join(os.path.dirname(__file__), '../corpus/')
corpus_file_path = os.path.join(corpus_directory, 'notion_corpus.json')

with open(corpus_file_path, 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)
