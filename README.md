# Pagerank Algorithm

Implementation in Python of the PageRank algorithm by Sampling (Markov Chain) and Power Iteration (Linear Algebra) methods, developed as an academic project.

## Overview 

The program uses the [Random Surfing Model](https://en.wikipedia.org/wiki/Random_surfing_model) to simulate the probability of a user visiting a web page in a corpus, resulting in a list of weights for each page based on its importance. 

While `sample_pagerank` uses a method built on probability, simulating each surfer's step, `iterate_pagerank` uses the convergence of the [Google Matrix](https://en.wikipedia.org/wiki/Google_matrix) eigenvector, making it a deterministic approach.

## How to Run
```
python pagerank.py [corpus_folder]
```

Note: The project already has two examples of corpora (`corpus1` and `corpus3`) for functional testing.

# Implementation Details

- **Sampling Method**: You can see how the damp factor influences the result by changing the `DAMPING` value to a number between 0 and 1. You can also change the number of `SAMPLES` to get a more or less precise outcome.

- **Power Iteration**: The `PRECISION` limit determines the convergence point of the eigenvector, allowing for a balance between performance and accuracy.