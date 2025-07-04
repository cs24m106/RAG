import re

def custom_text_splitter(text, chunk_size, chunk_overlap, word_split=False):
    """
    Splits a given text into chunks of a specified size with a defined overlap between them.

    This function divides the input text into chunks based on the specified chunk size and overlap.
    Optionally, it can split the text at word boundaries to avoid breaking words when 'word_split'
    is set to True. This is achieved by using a regular expression that identifies word separators.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of characters of overlap between consecutive chunks.
        word_split (bool, optional): If True, ensures that chunks end at word boundaries. Defaults to False.

    Returns:
        list of str: A list containing the text chunks.
    """
    chunks = []
    start = 0
    separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')
    
    while start < len(text) - chunk_overlap:
        end = min(start + chunk_size, len(text))
        
        if word_split:
            match = separators_pattern.search(text, end)
            if match:
                end = match.end()
                
        if end == start:
            end = start + 1
        
        sub = text[start:end]
        # Remove any whitespace (spaces, tabs, etc.) between newlines,
        chunk = re.sub(r'[ \t\r\f\v]*\n[ \t\r\f\v]*', '\n', sub)
        # then collapse multiple newlines to a single newline
        chunk = re.sub(r'\n+', '\n', chunk).strip()
        chunks.append(chunk)

        start = end - chunk_overlap
        
        if word_split:
            match = separators_pattern.search(text, start-1)
            if match:
                start = match.start() + 1
                
        if start < 0:
            start = 0
    
    return chunks


def chunk_doc(doc):
    """
    Each doc in split into chunks of default values:
    chunk_size = 500, chunk_overlap = 25, word_split = True
    
    returns a list of chunked entries in same dict format as of doc_entry
    => [chunk_entry -> each : {'text': chunk_content, 'source': doc_file_name}]
    """
    chunks= custom_text_splitter( doc["text"], 500, 25, word_split = True)
    return [{"text": chunk, "source": doc["source"]} for chunk in chunks]
 