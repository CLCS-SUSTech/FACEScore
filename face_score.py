import torch
import torch.nn as nn
import traceback
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from fft_utils import FFTProcessor
from metrics import cal_metrics


class FACEScorer:
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = 'cuda:0', max_length = 1024, batch_size=4, metrics=None, use_max=False):
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

    def load_model(self, model_path: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return model

    def init_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @torch.no_grad()
    def score_text(self, srcs: List[str], tgts: List[str], batch_size=None, fft_args=None):
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
                                                padding=True,
                                                max_length=self.max_length).to(self.device)
                tgt_encoded = self.tokenizer(tgt_list, 
                                                return_tensors='pt', 
                                                truncation=True, 
                                                padding=True,
                                                max_length=self.max_length).to(self.device)
                scores = self.score_encoded(src_encoded, tgt_encoded, fft_args)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'src_list: {src_list}')
                print(f'tgt_list: {tgt_list}')
                exit(0)
            final_scores.extend(scores)
        return final_scores
    
    @torch.no_grad()
    def get_spectrum_from_text(self, texts: List[str], batch_size=None, fft_args=None):
        if batch_size is None:
            batch_size = self.batch_size
        nll_list = []
        for i in range(0, len(texts), batch_size):
            text_list = texts[i:i+batch_size]
            try:
                encoded = self.tokenizer(text_list, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        max_length=self.max_length).to(self.device)
                nlls = self.get_nll(encoded)
                nll_list.extend(nlls)
                
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'text_list: {text_list}')
                exit(0)

        df = self.get_spectrum(nll_list, fft_args, packed=True)
        return df

    def score_encoded(self, src_encoded, tgt_encoded, fft_args=None):
        src_nll = self.get_nll(src_encoded)
        tgt_nll = self.get_nll(tgt_encoded)
        scores = self.score_nlls(src_nll, tgt_nll, fft_args)
        return scores
    
    def score_nlls(self, src_nll: List, tgt_nll: List, fft_args=None):
        src_powers, src_freqs = self.get_spectrum(src_nll, fft_args)
        tgt_powers, tgt_freqs = self.get_spectrum(tgt_nll, fft_args)
        scores = self.spectrum_dist(src_powers, src_freqs, tgt_powers, tgt_freqs, self.metrics)
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
    
    def get_spectrum(self, nlls, fft_args=None, packed=False):
        nlls = [nll.cpu().numpy() for nll in nlls]
        if not self.use_max:
            nlls = [(nll[:1000] if len(nll) > 1000 else nll) for nll in nlls]
        if fft_args is None:
            fft_processor = FFTProcessor()
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
