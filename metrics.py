import numpy as np
from scipy.stats import entropy, zscore, spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d


def spectral_overlap(spectrum1, spectrum2):
    spectrums = np.abs(np.concatenate([spectrum1[None], spectrum2[None]], axis=0))
    return spectrums.min(axis=0).sum() / spectrums.max(axis=0).sum()

def pearson_correlation(spectrum1, spectrum2):
    return np.corrcoef(spectrum1, spectrum2)[0, 1]

def spearman_correlation(spectrum1, spectrum2):
    corr, _ = spearmanr(spectrum1, spectrum2)
    return corr

def earth_mover_distance(spectrum1, spectrum2):
    p1 = [spectrum1[0]]
    p2 = [spectrum2[0]]
    for value in spectrum1[1:]:
        p1.append(p1[-1] + value)
    for value in spectrum2[1:]:
        p2.append(p2[-1] + value)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p1 /= p1[-1]
    p2 /= p2[-1]
    return np.abs(p1 - p2).sum() / spectrum1.shape[0]

def kl_divergence(spectrum1, spectrum2): 
    # NOTICE: KL(spectrum2||spectrum1), the first spectrum is taken as the groundtruth
    # NOTICE: spectrum will be normalized in function
    # find where the spectrum1 is 0, and add a small value to avoid log(0)
    spectrum1 = np.where(spectrum1 == 0, 1e-10, spectrum1)
    return entropy(spectrum2, spectrum1)

def jensen_shannon_divergence(spectrum1, spectrum2):
    # NOTICE: spectrum will be normalized in function
    return jensenshannon(spectrum1, spectrum2)

def cal_metrics(spectrum1, freqs1, spectrum2, freqs2, metrics=None, use_max=False):
    if metrics is None or len(metrics) == 0:
        metrics = ['so', 'corr', 'spearman', 'emd', 'kl', 'js']
    results = []
    for sp1, freq1, sp2, freq2 in zip(spectrum1, freqs1, spectrum2, freqs2):
        if use_max:
            length = max(len(sp1), len(sp2))
        else:
            length = 1000
            
        x = np.linspace(0, 0.5, length)

        x_1 = np.linspace(0, 0.5, len(sp1))
        x_2 = np.linspace(0, 0.5, len(sp2))
        f1 = interp1d(x_1, sp1, kind='linear', fill_value='extrapolate')
        f2 = interp1d(x_2, sp2, kind='linear', fill_value='extrapolate')

        # f1 = interp1d(freq1, sp1, kind='linear', fill_value='extrapolate')
        # f2 = interp1d(freq2, sp2, kind='linear', fill_value='extrapolate')
        
        sp1 = f1(x)
        sp2 = f2(x)

        result = {}
        for metric in metrics:
            if metric == 'so':
                result['so'] = spectral_overlap(sp1, sp2)
            elif metric == 'corr':
                result['corr'] = pearson_correlation(sp1, sp2)
            elif metric == 'spearman':
                result['spear'] = spearman_correlation(sp1, sp2)
            elif metric == 'emd':
                result['emd'] = earth_mover_distance(sp1, sp2)
            elif metric == 'kl':
                result['kl'] = kl_divergence(sp1, sp2)
            elif metric == 'js':
                result['js'] = jensen_shannon_divergence(sp1, sp2)
        results.append(result)
        
    return results


