# FACEScore

This repo contains code for FACEScore (FACE for *F*ourier *A*nalysis of *C*ross-*E*ntropy), a metric for evaluating open-ended natural language genration (NLG) using spectral features of text surprisal. 


### Direct Use

```python
from face_score import FACEScorer

model_path = 'path/to/your/model' # For instance, 'models/gpt2'
scorer = FACEScorer(model_path, device='cpu')

texts1 = ["Hello, world!", "How are you?", "Goodbye, world!", "I am fine."]
texts2 = ["Hello, hello, world!", "How do you do?", "See you around, I say.", "I am fine, thank you."]

print(scorer.score_texts(texts1, texts2))

# use other fft args
scorer.fft_preprocess = 'zscore'
scorer.fft_value = 'real'
print(scorer.score_texts(texts1, texts2))
```