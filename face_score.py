import numpy as np
import torch
import torch.nn as nn
import traceback
from typing import List, Iterable
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from fft_utils import FFTProcessor
from metrics import cal_metrics


class FACEScorer:
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = 'cuda:0', 
                 max_length = 1024, batch_size=4, metrics=None, use_max=False,
                 fft_args=None):
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
        self.metrics = metrics
        self.use_max = use_max
        if fft_args is None:
            self.fft_processor = FFTProcessor()
        else:
            self.fft_processor = FFTProcessor(method=fft_args['method'] if 'method' in fft_args else 'fft',
                                        preprocess=fft_args['preprocess'] if 'preprocess' in fft_args else 'none',
                                        value=fft_args['value'] if 'value' in fft_args else 'norm',
                                        require_sid=fft_args['require_sid'] if 'require_sid' in fft_args else True,
                                        verbose=fft_args['verbose'] if 'verbose' in fft_args else False)

    def load_model(self, model_path: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return model

    def init_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @torch.no_grad()
    def score_texts(self, srcs: List[str], tgts: List[str], batch_size=None, fft_args=None):
        assert len(srcs) == len(tgts)
        if batch_size is None:
            batch_size = self.batch_size
        final_scores = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]
            try:
                src_encoded = self.texts_to_encoded(src_list)
                tgt_encoded = self.texts_to_encoded(tgt_list)
                scores = self.score_encoded(src_encoded, tgt_encoded, fft_args)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'src_list: {src_list}')
                print(f'tgt_list: {tgt_list}')
                exit(0)
            final_scores.extend(scores)
        return final_scores
    
    def score_encoded(self, src_encoded, tgt_encoded, fft_args=None):
        src_nll = self.encoded_to_nll(src_encoded)
        tgt_nll = self.encoded_to_nll(tgt_encoded)
        scores = self.score_nlls(src_nll, tgt_nll, fft_args)
        return scores
    
    def score_nlls(self, src_nll: List, tgt_nll: List, fft_args=None):
        src_powers, src_freqs = self.nll_to_spectrum(src_nll, fft_args)
        tgt_powers, tgt_freqs = self.nll_to_spectrum(tgt_nll, fft_args)
        scores = self.spectrum_dist(src_powers, src_freqs, tgt_powers, tgt_freqs, self.metrics)
        return scores
    
    @torch.no_grad()
    def texts_to_encoded(self, texts: List[str], batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size >= len(texts):
            try:
                encoded = self.tokenizer(texts, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        max_length=self.max_length).to(self.device)
            except RuntimeError:
                traceback.print_exc()
                print(f'running tokenizer failed')
                print(f'texts: {texts}')
                exit(0)
            return encoded
        else:
            return list(self.texts_to_encoded_iter(texts, batch_size))

    @torch.no_grad()
    def texts_to_encoded_iter(self, texts: List[str], batch_size=None) -> Iterable:
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
    def encoded_to_nll(self, encoded) -> List:
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
    
    
    
    def nll_to_spectrum(self, nlls: List, fft_args=None, packed=False):
        if isinstance(nlls[0], torch.Tensor):
            nlls = [nll.cpu().numpy() for nll in nlls]
        elif isinstance(nlls[0], list):
            nlls = [np.array(nll) for nll in nlls]

        if not self.use_max:
            nlls = [(nll[:1000] if len(nll) > 1000 else nll) for nll in nlls]

        if fft_args is None:
            fft_processor = self.fft_processor
        else:
            fft_processor = FFTProcessor(method=fft_args['method'] if 'method' in fft_args else 'fft',
                                        preprocess=fft_args['preprocess'] if 'preprocess' in fft_args else 'none',
                                        value=fft_args['value'] if 'value' in fft_args else 'norm',
                                        require_sid=fft_args['require_sid'] if 'require_sid' in fft_args else True,
                                        verbose=fft_args['verbose'] if 'verbose' in fft_args else False)
        if packed:
            df = fft_processor.process(nlls, packed=True)
            return df
        else:
            freqs, powers, _ = fft_processor.process(nlls, packed=False)
            return powers, freqs
    
    def spectrum_dist(self, src_p, src_f, tgt_p, tgt_f, metrics=None):
        results = cal_metrics(src_p, src_f, tgt_p, tgt_f, metrics, self.use_max)
        return results
    

    @torch.no_grad()
    def texts_to_nll(self, texts: List[str], batch_size=None) -> List:
        """
        For quick experiment over a text input
        """
        if batch_size is None:
            batch_size = self.batch_size
        nll_list = []
        for encoded in tqdm(self.texts_to_encoded_iter(texts, batch_size), total=len(texts)//batch_size+1):
            nll_list.extend(self.encoded_to_nll(encoded))
        return nll_list

    @torch.no_grad()
    def texts_to_spectrum(self, texts: List[str], batch_size=None, fft_args=None):
        """
        For quick experiment over a text input
        """
        nll_list = self.texts_to_nll(texts, batch_size)
        df = self.nll_to_spectrum(nll_list, fft_args, packed=True)
        return df
