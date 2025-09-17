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
    def __init__(self, 
                 model_path: str = None,
                 tokenizer_path: str = None, 
                 device: str = 'cuda:0', 
                 max_length = 1024, 
                 batch_size=4, 
                 metrics=None, 
                 fft_method='fft',
                 fft_preprocess='none',
                 fft_value='norm',
                 fft_require_sid=True,
                 fft_verbose=False,
                 use_max=False,
                 save_intermediate=True):
        """
        :param model_path: path to the model, None if inferencing not needed
        :param tokenizer_path: path to the tokenizer, if None, tokenizer will be loaded from model_path
        :param device: cuda device to run the model
        :param max_length: max length of the input text
        :param batch_size: batch size for processing
        :param metrics: a list of metrics to calculate the distance between two spectra, choose from ['so', 'corr', 'spearman', 'emd', 'kl', 'js']
        :param fft_method: 'fft' or 'periodogram'
        :param fft_preprocess: 'none', 'zscore', 'minmax', 'log', 'logzs
        :param fft_value: 'norm', 'real', 'imag'
        :param fft_require_sid: whether to output the sids of the spectrums
        :param fft_verbose: whether to print the processing details
        :param use_max: whether to use the max length of the two spectrums to do the interpolation, if False, use 1000 as the length
        """
        if model_path is not None:
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
        self.save_intermediate = save_intermediate
        if save_intermediate:
            self.intermediates = {
                'src_nlls': [],
                'tgt_nlls': [],
                'src_powers': [],
                'tgt_powers': [],
                'src_freqs': [],
                'tgt_freqs': [],
            }

        self.fft_processor = FFTProcessor(method=fft_method,
                                        preprocess=fft_preprocess,
                                        value=fft_value,
                                        require_sid=fft_require_sid,
                                        verbose=fft_verbose)

        self.fft_method = fft_method
        self.fft_preprocess = fft_preprocess
        self.fft_value = fft_value
        self.fft_require_sid = fft_require_sid
        self.fft_verbose = fft_verbose
        

    def load_model(self, model_path: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return model

    def init_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @torch.no_grad()
    def score_texts(self, srcs: List[str], tgts: List[str], dist_name: str = 'emd', batch_size=None):
        assert len(srcs) == len(tgts)
        if batch_size is None:
            batch_size = self.batch_size
        all_results = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]
            try:
                src_encoded = self.texts_to_encoded(src_list)
                tgt_encoded = self.texts_to_encoded(tgt_list)
                scores = self.score_encoded(src_encoded, tgt_encoded)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'src_list: {src_list}')
                print(f'tgt_list: {tgt_list}')
                exit(0)
            all_results.extend(scores)
        # save results
        self.all_results = all_results 
        # collect 
        results = self.collect(dist_name)
        return results
    
    def score_encoded(self, src_encoded, tgt_encoded):
        src_nll = self.encoded_to_nll(src_encoded)
        tgt_nll = self.encoded_to_nll(tgt_encoded)
        scores = self.score_nlls(src_nll, tgt_nll)
        if self.save_intermediate:
            self.intermediates['src_nlls'].append(src_nll)
            self.intermediates['tgt_nlls'].append(tgt_nll)
        return scores
    
    def score_nlls(self, src_nll: List, tgt_nll: List):
        src_powers, src_freqs = self.nll_to_spectrum(src_nll)
        tgt_powers, tgt_freqs = self.nll_to_spectrum(tgt_nll)
        scores = self.spectrum_dist(src_powers, src_freqs, tgt_powers, tgt_freqs)
        if self.save_intermediate:
            self.intermediates['src_powers'].append(src_powers)
            self.intermediates['tgt_powers'].append(tgt_powers)
            self.intermediates['src_freqs'].append(src_freqs)
            self.intermediates['tgt_freqs'].append(tgt_freqs)
        return scores
    
    @torch.no_grad()
    def texts_to_encoded(self, texts: List[str], batch_size=None):
        assert self.tokenizer is not None
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size >= len(texts):
            try:
                encoded = self.tokenizer(texts, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True,
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
        assert self.tokenizer is not None
        if batch_size is None:
            batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            text_list = texts[i:i+batch_size]
            try:
                encoded = self.tokenizer(text_list, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True,
                                        max_length=self.max_length).to(self.device)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'text_list: {text_list}')
                exit(0)
            yield encoded

    @torch.no_grad()
    def encoded_to_nll(self, encoded) -> List:
        assert self.model is not None
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
    
    def spectrum_dist(self, src_p, src_f, tgt_p, tgt_f):
        results = cal_metrics(src_p, src_f, tgt_p, tgt_f, self.metrics, self.use_max)
        return results
    
    def collect(self, dist_name:str='emd'):
        if self.all_results is None:
            raise ValueError('No results to collect')
        collected = []
        if dist_name in ['so', 'corr', 'spearman', 'emd', 'kl', 'js']:
            for res in self.all_results:
                collected.append(res[dist_name])
        elif dist_name == 'ensemble3':
            pass
        elif dist_name == 'ensemble5':
            pass
        else:
            raise ValueError(f'Invalid dist_name: {dist_name}')
        return collected

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
    def texts_to_spectrum(self, texts: List[str], batch_size=None):
        """
        For quick experiment over a text input
        """
        nll_list = self.texts_to_nll(texts, batch_size)
        df = self.nll_to_spectrum(nll_list, packed=True)
        return df
