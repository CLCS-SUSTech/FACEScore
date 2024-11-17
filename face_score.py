import torch
import torch.nn as nn
import traceback
from typing import List


class FACEScorer:
    def __init__(self, model, device, max_length):
        self.tokenizer = None
        pass

    def score_text(self, srcs: List[str], tgts: List[str], batch_size=4):
        assert len(srcs) == len(tgts)
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]
            try:
                with torch.no_grad():
                    # src_encoded = self.tokenizer(src_list, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
                    # tgt_encoded = self.tokenizer(tgt_list, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
                    # scores = self.score_encoded(src_encoded, tgt_encoded)
                    pass
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'src_list: {src_list}')
                print(f'tgt_list: {tgt_list}')
                exit(0)

    def score_encoded(self, src_encoded, tgt_encoded):
        pass