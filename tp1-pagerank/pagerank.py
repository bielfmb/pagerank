import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000
PRECISION = 0.001


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    p = dict()
    links = corpus[page]

    # Uniform distribution if there are no links in `page`
    if not links:
        p_teleport = 1/len(corpus)
        for file in corpus:
            p[file] = p_teleport

        return p
    
    p_teleport = (1 - damping_factor) * (1/len(corpus))
    p_link = damping_factor * (1/len(links))
    
    for file in corpus:
        p[file] = p_teleport

        if file in links:
            p[file] += p_link

    return p


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 0 for page in corpus}
    curr = random.choice(list(corpus.keys()))

    for _ in range(n):  
        ranks[curr] += 1

        p_dist = transition_model(corpus, curr, damping_factor)
        curr = random.choices(list(p_dist.keys()), 
                              weights=list(p_dist.values()), 
                              k=1
                              )[0]

    for p in ranks:
        ranks[p] /= n

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)

    E = np.ones((n, n))
    L = get_transition_matrix(corpus)
    Google_Matrix = ((1 - damping_factor)/n) * E + damping_factor * L

    # Initial uniform distribution for `p`
    p = np.full(n, 1/n)

    while True:
        last_p = p.copy()

        p = Google_Matrix @ p

        if (np.max(np.abs(p - last_p)) < PRECISION):
            break

    return dict(zip(corpus.keys(), p))


def get_transition_matrix(corpus):
    """
    Return a transition matrix `M` by verifying the existence of 
    a link from a `page` j to an i. If so, the element `M_{i,j}`
    represents the probability of taking that link.

    If there are no links coming from `page` j, the whole column
    receives `1/number of pages in the corpus`, otherwise it
    gets `1/number of links coming from the `page` j`.
    """
    pages = list(corpus.keys())
    n = len(pages)
    M = np.zeros((n, n))

    for j, from_page in enumerate(pages):
        links = corpus[from_page]

        if not links:
            M[:, j] = 1 / n

        else:
            for i, to_page in enumerate(pages):
                if to_page in links:
                    M[i, j] = 1 / len(links)

    return M


if __name__ == "__main__":

    main()
