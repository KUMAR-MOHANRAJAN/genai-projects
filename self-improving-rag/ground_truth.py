"""Ground truth: test queries with expected answers and retrieval keywords.

Used by:
  - Evaluator: compare pipeline answer to expected_answer (correctness scoring)
  - Retrieval metrics: check if retrieved chunks contain keywords (precision/recall proxy)
  - Optimizer loop: the target the system optimizes toward

Keywords are strategy-agnostic — they work across fixed_size, recursive_split,
and semantic chunking without needing per-version chunk ID mapping.
"""

TEST_QUERIES = [
    (
        "How do embeddings represent meaning?",
        "Embeddings map text to dense numerical vectors where semantically similar "
        "texts have similar vector representations. Words or sentences with related "
        "meanings end up close together in the vector space.",
        ["embedding", "vector", "semantic", "similarity", "dense"],
    ),
    (
        "What is the attention mechanism?",
        "Attention weighs the importance of each token relative to all other tokens "
        "in a sequence. It allows the model to focus on relevant parts of the input "
        "when processing each position.",
        ["attention", "token", "weight", "sequence", "focus"],
    ),
    (
        "What is a transformer?",
        "A transformer is a neural network architecture that uses self-attention "
        "to process sequences in parallel, rather than sequentially like RNNs. "
        "It consists of encoder and decoder stacks with attention and feed-forward layers.",
        ["transformer", "attention", "parallel", "sequence", "neural"],
    ),
    (
        "What is tokenization?",
        "Tokenization splits text into smaller units called tokens that the model "
        "can process. Tokens can be words, subwords, or characters depending on "
        "the tokenizer used.",
        ["token", "split", "text", "unit", "subword"],
    ),
    (
        "What is a context window?",
        "The context window is the maximum number of tokens the model can process "
        "in a single forward pass. It limits how much text the model can see at once.",
        ["context", "window", "token", "maximum", "process"],
    ),
    (
        "What is fine-tuning?",
        "Fine-tuning adapts a pre-trained model to a specific task by continuing "
        "training on domain-specific data. It adjusts the model's weights to perform "
        "better on the target task.",
        ["fine-tuning", "pre-trained", "task", "training", "weights"],
    ),
    (
        "How does word2vec generate word embeddings?",
        "Word2vec generates word embeddings using neural networks that predict "
        "whether two words are neighbors in a sentence. Each word starts with a "
        "random vector embedding and the vectors are updated during training so "
        "that words appearing in similar contexts end up with similar embeddings.",
        ["word2vec", "embedding", "neural", "neighbor", "training"],
    ),
]


def load_ground_truth() -> list[tuple[str, str, list[str]]]:
    """Return list of (question, expected_answer, keywords) tuples."""
    return TEST_QUERIES


def get_query(index: int = 0) -> tuple[str, str, list[str]]:
    """Get a specific test query by index."""
    return TEST_QUERIES[index]
