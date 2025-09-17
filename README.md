# FACEScore

This repo contains code for FACEScore (FACE for *F*ourier *A*nalysis of *C*ross-*E*ntropy), a metric for evaluating open-ended natural language genration (NLG) using spectral features of text surprisal. 

### Installation

```bash
# Install from source
git clone https://github.com/CLCS-SUSTech/FACEScore.git
cd FACEScore
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Usage

```python
from face_score import FACEScorer

model_path = 'path/to/your/model' # For instance, 'models/gpt2'
scorer = FACEScorer(model_path, device='cpu')

texts1 = ["Hello, world!", "How are you?", "Goodbye, world!", "I am fine."]
texts2 = ["Hello, hello, world!", "How do you do?", "See you around, I say.", "I am fine, thank you."]

results = scorer.score_texts(texts1, texts2, dist_name='emd')
print(results)
```

The output would be: 

```
[np.float64(0.07721481542937722), np.float64(0.024251873990435706), np.float64(0.09032722449768522), np.float64(0.027596788635347518)]
```

Once the calculation is done, and you can collect the scores from other distance functions:

```python
scorer.collect(dist_name='so')
```

which produces a different of scores:
```
[np.float64(0.66378198918476), np.float64(0.7913692274742679), np.float64(0.5903860651767472), np.float64(0.8581959053156671)]
```

You can change the arguments for Fourier transform:

```
# use other fft args
scorer.fft_preprocess = 'zscore'
scorer.fft_value = 'real'
```