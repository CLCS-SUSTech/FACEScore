import torch
import torch.nn as nn
import traceback
from typing import List


class FACEScorer:
    def __init__(self, model, device, max_length):
        self.tokenizer = None
        self.nll_loss = nn.NLLLoss(reduce='none')
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.spectrum_dist = None
        pass

    def score_text(self, srcs: List[str], tgts: List[str], batch_size=4):
        assert len(srcs) == len(tgts)
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]
            try:
                with torch.no_grad():
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
            return scores

    def score_encoded(self, src_encoded, tgt_encoded):
        src_nll = self.get_nll(src_encoded)
        tgt_nll = self.get_nll(tgt_encoded)
        src_spectrum = self.get_spectrum(src_nll)
        tgt_spectrum = self.get_spectrum(tgt_nll)
        scores = self.spectrum_dist(src_spectrum, tgt_spectrum)
        return scores

    def get_nll(self, encoded):
        ids = encoded['input_ids']
        output = self.model(ids, labels=ids)
        logits = output.logits.to(self.device)
        logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
        shift_logits = logits[:, :, :-1]
        shift_targets = ids[:, 1:]
        nlls = self.nll_loss(self.log_softmax(shift_logits), shift_targets)
        # post processing
        return nlls