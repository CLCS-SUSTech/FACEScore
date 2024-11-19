import torch
import torch.nn as nn
import traceback
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


class FACEScorer:
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = 'cuda:0', max_length = 1024, batch_size=4):
        self.model = self.load_model(model_path, device)
        if tokenizer_path is None:
            self.tokenizer = self.init_tokenizer(model_path)
        else:
            self.tokenizer = self.init_tokenizer(tokenizer_path)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.log_softmax = nn.LogSoftmax(dim=1)

    def load_model(self, model_path: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return model

    def init_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @torch.no_grad()
    def score_text(self, srcs: List[str], tgts: List[str], batch_size=None):
        assert len(srcs) == len(tgts)
        if batch_size is None:
            batch_size = self.batch_size
        final_scores = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]
            try:
                src_encoded = self.tokenizer(src_list, 
                                                return_tensors='pt', 
                                                truncation=True, 
                                                max_length=self.max_length).to(self.device)
                tgt_encoded = self.tokenizer(tgt_list, 
                                                return_tensors='pt', 
                                                truncation=True, 
                                                max_length=self.max_length).to(self.device)
                scores = self.score_encoded(src_encoded, tgt_encoded)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'src_list: {src_list}')
                print(f'tgt_list: {tgt_list}')
                exit(0)
            final_scores.extend(scores)
        return final_scores

    def score_encoded(self, src_encoded, tgt_encoded):
        src_nll = self.get_nll(src_encoded)
        tgt_nll = self.get_nll(tgt_encoded)
        src_spectrum = self.get_spectrum(src_nll)
        tgt_spectrum = self.get_spectrum(tgt_nll)
        scores = self.spectrum_dist(src_spectrum, tgt_spectrum)
        return scores
    
    @torch.no_grad()
    def get_encoded(self, texts: List[str], batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            text_list = texts[i:i+batch_size]
            try:
                encoded = self.tokenizer(text_list, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        max_length=self.max_length).to(self.device)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'text_list: {text_list}')
                exit(0)
            yield encoded

    @torch.no_grad()
    def get_nll(self, encoded) -> List:
        ids = encoded['input_ids']
        output = self.model(ids, labels=ids)
        logits = output.logits.to(self.device)
        logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
        shift_logits = logits[:, :, :-1]
        shift_targets = ids[:, 1:]

        nlls = self.nll_loss(self.log_softmax(shift_logits), shift_targets)
        mask = encoded['attention_mask'][:, 1:]
        nll_list = []
        for i in range(nlls.shape[0]): # Along B dimension
            raw = nlls[i, :]
            nll = torch.masked_select(raw, mask[i, :]>0)
            nll_list.append(nll)

        return nll_list
    
    def save_nll(self, nlls: List, path: str, decimal=4):
        with open(path, 'w') as f:
            for nll in nlls:
                if isinstance(nll, torch.Tensor):
                    nll = nll.tolist()
                f.write(' '.join([f'{x:.{decimal}f}' for x in nll]) + '\n')
    
    def get_spectrum(self, nlls):
        #TODO: post processing
        return nlls
    
    def spectrum_dist(self, src_sp, tgt_sp, dist_func: str = 'EMD'):
        #TODO: post processing
        return torch.tensor(0)