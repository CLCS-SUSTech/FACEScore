from typing import List

def load_text_data(path: str) -> List[str]:
    """
    Load text data from a file.
    Args:
        path (str): Path to the file containing text data.
    Returns:
        List[str]: List of text data.
    """
    with open(path, 'r') as f:
        data = f.readlines()
    return data