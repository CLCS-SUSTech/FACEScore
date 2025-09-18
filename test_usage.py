#!/usr/bin/env python3
"""
Test script for FACEScore usage example from README.md
"""

from face_score import FACEScorer

# Test the basic usage example
model_path = '/Users/xy/models/gpt2'
scorer = FACEScorer(model_path, device='cpu')

texts1 = ["Hello, world!", "How are you?", "Goodbye, world!", "I am fine."]
texts2 = ["Hello, hello, world!", "How do you do?", "See you around, I say.", "I am fine, thank you."]

print("Testing FACEScore with the following texts:")
print("Texts1:", texts1)
print("Texts2:", texts2)
print("\nCalculating scores...")

try:
    scores = scorer.score_texts(texts1, texts2)
    print("Scores calculated successfully!")
    print("Results:", scores)
    
    # Test with different distance metrics
    print("\nTesting with different distance metrics:")
    for metric in ['so', 'corr', 'spearman', 'emd', 'kl', 'js']:
        try:
            metric_scores = scorer.collect(metric)
            print(f"{metric}: {metric_scores}")
        except Exception as e:
            print(f"Error with {metric}: {e}")
    
    # Check intermediate results if saved
    if hasattr(scorer, 'save_intermediate') and scorer.save_intermediate:
        print(f"\nIntermediate results saved: {len(scorer.intermediates['src_nlls'])} batches")
        print("Available intermediate data:", list(scorer.intermediates.keys()))
        
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
