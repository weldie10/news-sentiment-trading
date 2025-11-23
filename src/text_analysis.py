"""
Text analysis utilities for headline processing and NLP operations.

This module provides functions for preprocessing text, extracting n-grams,
identifying keywords, and finding financial-specific terms in headlines.

Example:
    >>> from src.text_analysis import preprocess_text, get_top_keywords
    >>> tokens = preprocess_text("Apple stock rises 5%")
    >>> keywords = get_top_keywords(df, 'headline', top_n=20)
"""
import re
import pandas as pd
from collections import Counter
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocess text for analysis by tokenizing and cleaning.
    
    Processing steps:
    1. Convert to lowercase
    2. Remove special characters (keep alphanumeric and spaces)
    3. Tokenize using NLTK word_tokenize
    4. Remove stopwords (if enabled)
    5. Filter tokens shorter than 3 characters
    
    Args:
        text: Input text string to preprocess
        remove_stopwords: Whether to remove English stopwords (default: True)
        
    Returns:
        List of processed tokens (strings)
        
    Example:
        >>> preprocess_text("Apple's stock rises 5%!")
        ['apple', 'stock', 'rises']
        >>> preprocess_text("The stock is rising", remove_stopwords=False)
        ['the', 'stock', 'is', 'rising']
    """
    # Handle missing values
    if pd.isna(text):
        return []
    
    # Step 1: Convert to lowercase for consistency
    text = str(text).lower()
    
    # Step 2: Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Step 3: Tokenize using NLTK
    tokens = word_tokenize(text)
    
    # Step 4: Remove stopwords if requested (common words like 'the', 'is', etc.)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Step 5: Filter out very short tokens (likely noise)
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text string
        n: Size of n-gram (2 for bigrams, 3 for trigrams)
        
    Returns:
        List of n-grams
    """
    tokens = preprocess_text(text, remove_stopwords=True)
    ngrams = []
    
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def get_top_keywords(df: pd.DataFrame, column: str = 'headline', 
                     top_n: int = 20, ngram_size: int = 1) -> pd.Series:
    """
    Get top keywords from headlines.
    
    Args:
        df: DataFrame with text data
        column: Column name containing text
        top_n: Number of top keywords to return
        ngram_size: Size of n-grams (1 for unigrams, 2 for bigrams, etc.)
        
    Returns:
        Series with top keywords and their counts
    """
    all_tokens = []
    
    for text in df[column]:
        if ngram_size == 1:
            tokens = preprocess_text(text)
        else:
            tokens = extract_ngrams(text, n=ngram_size)
        all_tokens.extend(tokens)
    
    counter = Counter(all_tokens)
    top_keywords = pd.Series(dict(counter.most_common(top_n)))
    
    return top_keywords


def extract_financial_keywords(df: pd.DataFrame, column: str = 'headline') -> Dict[str, int]:
    """
    Extract financial-specific keywords from headlines.
    
    Args:
        df: DataFrame with headlines
        column: Column name containing headlines
        
    Returns:
        Dictionary with financial keywords and their counts
    """
    financial_terms = [
        'price target', 'fda approval', 'earnings', 'eps', 'revenue',
        'maintains', 'raises', 'lowers', 'upgrade', 'downgrade',
        'buy', 'sell', 'hold', 'overweight', 'underweight',
        'analyst', 'rating', 'stock', 'shares', 'trading',
        'market', 'quarter', 'q1', 'q2', 'q3', 'q4'
    ]
    
    keyword_counts = {}
    
    for term in financial_terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        count = df[column].str.contains(pattern, na=False).sum()
        if count > 0:
            keyword_counts[term] = count
    
    return keyword_counts

