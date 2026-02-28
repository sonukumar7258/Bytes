import os

def custom_chunking_website(directory: str = 'website_data') -> list:
    """
    Chunk website data from text files into a list of strings.

    Args:
        directory (str, optional): The directory containing the text files. Defaults to 'website_data'.

    Returns:
        list: A list of chunked website data as strings.
    """
    chunks = []
    arbitrary_strings_to_remove = [
        "Expertise", "Other Services", "Web 3", "EdTech", "Commercial Driving", "Security", "Finance", 
        "On-Demand Services", "Automation", "Services", "Work", "Web 3", "EdTech", "Automation", "Platforms"
    ]

    # Iterate over text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
            # Construct a URL from the filename
            url = 'https://' + filename.replace('_', '/').replace('.txt', '')
            # Modify the lines by adding the URL and removing unwanted strings
            modified_lines = [url + '\n'] + lines
            modified_lines = [line for line in modified_lines if line.strip() not in arbitrary_strings_to_remove]
            # Join the modified lines into a single chunk
            chunk = ''.join(modified_lines)
            chunk = 'website\n' + chunk
            # Append the chunk to the list
            chunks.append(chunk)
    return chunks