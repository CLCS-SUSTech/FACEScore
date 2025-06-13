from typing import List
from face_score import FACEScorer
from tqdm import tqdm
import torch

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

def test_FACEScorer():
    scorer = FACEScorer("gpt2", device='cpu')
    
    # Test case 1: Test with a single source and target text
    srcs1 = ["Hello, world!"]
    tgts1 = ["Hello, world!"]
    scores1 = scorer.score_texts(srcs1, tgts1)
    assert isinstance(scores1, torch.Tensor)

    # Test case 2: Test with multiple source and target texts
    srcs2 = ["Hello, world!", "How are you?"]
    tgts2 = ["Hello, world!", "I am fine."]
    scores2 = scorer.score_texts(srcs2, tgts2)
    assert isinstance(scores2, torch.Tensor)

    # Test case 3: Test with source and target texts of different lengths
    srcs5 = ["Hello, world!"]
    tgts5 = ["Hello, world! How are you?"]
    scores5 = scorer.score_texts(srcs5, tgts5)
    assert isinstance(scores5, torch.Tensor)

    print("All test cases pass")


def test_get_nll():
    model_path = '/Users/xy/models/gpt2'
    scorer = FACEScorer(model_path, device='cpu')

    text1 = ["Hello, world!", "How are you?", "Goodbye, world!", "I am fine."]
    text2 = ["Hello, hello, world!", "How do you do?", "See you around, I say.", "I am fine, thank you."]
    enc1 = scorer.tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    enc2 = scorer.tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
    # print(text1_enc)
    # print(text2_enc)

    # nll1 = scorer.get_nll(enc1)
    # nll2 = scorer.get_nll(enc2)
    # print('nll1:', nll1.shape)
    # # print('nll2:', nll2)

    # mask1 = enc1['attention_mask']
    # print('mask1:', mask1.shape)
    # print('mask1[:,1:]:', mask1[:,1:].shape)
    # nll1m = nll1 * mask1[:,1:]
    # print('nll1m:', nll1m.shape)

    # print()
    # print('Before mask, nll1 = ', nll1)
    # print('After mask, nll1m = ', nll1m)

    nll_list1 = scorer.encoded_to_nll(enc1)
    print(nll_list1)


def test_pythia_nll():
    model_path = '/data1/model/pythia-410m-base'
    scorer = FACEScorer(model_path, device='cuda:0')
    # scorer = FACEScorer(model_path, device='cpu')

    text_file = 'sample/sample_text_bbc_gemma2-2b.txt'
    texts = load_text_data(text_file)

    final_nlls = []
    for enc in tqdm(scorer.texts_to_encoded(texts)):
        nll_list = scorer.encoded_to_nll(enc)
        final_nlls.extend(nll_list)
    scorer.save_nll(final_nlls, 'sample/sample_nll_bbc_gemma2-2b_out.txt')    

# test_FACEScorer()
# test_get_nll()
test_pythia_nll()
