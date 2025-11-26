"""
Sentiment analysis for financial news headlines.

This module provides a class-based interface for analyzing sentiment
of financial news using multiple NLP libraries (NLTK, TextBlob, VADER).

Example:
    >>> analyzer = SentimentAnalyzer()
    >>> df['sentiment'] = analyzer.analyze_sentiment(df['headline'])
    >>> daily_sentiment = analyzer.aggregate_daily_sentiment(df)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from textblob import TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER not available. Install with: pip install vaderSentiment")

try:
    from .text_analysis import TextAnalyzer
except ImportError:
    try:
        from text_analysis import TextAnalyzer
    except ImportError:
        # TextAnalyzer not strictly required for sentiment analysis
        TextAnalyzer = None


class SentimentAnalyzer:
    """
    A class for analyzing sentiment of financial news headlines.
    
    Uses multiple sentiment analysis methods (TextBlob, VADER) and
    provides aggregation capabilities for daily sentiment scores.
    
    Attributes:
        vader_analyzer: VADER sentiment analyzer instance
        text_analyzer: TextAnalyzer instance for preprocessing
        
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> df['sentiment_score'] = analyzer.analyze_sentiment(df['headline'])
        >>> df['sentiment_label'] = analyzer.classify_sentiment(df['sentiment_score'])
    """
    
    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        self.text_analyzer = TextAnalyzer() if TextAnalyzer is not None else None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
    
    def analyze_sentiment_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob.
        
        Returns polarity score ranging from -1 (negative) to 1 (positive).
        
        Args:
            text: Text string to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1)
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> score = analyzer.analyze_sentiment_textblob("Stock rises 5%")
            >>> print(f"Sentiment: {score:.2f}")
        """
        if pd.isna(text):
            return 0.0
        
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    
    def analyze_sentiment_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER.
        
        Returns compound score ranging from -1 (negative) to 1 (positive).
        VADER is particularly effective for social media and news text.
        
        Args:
            text: Text string to analyze
            
        Returns:
            VADER compound sentiment score (-1 to 1)
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> score = analyzer.analyze_sentiment_vader("Great earnings report!")
            >>> print(f"VADER sentiment: {score:.2f}")
        """
        if not VADER_AVAILABLE or self.vader_analyzer is None:
            return 0.0
        
        if pd.isna(text):
            return 0.0
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        return scores['compound']
    
    def analyze_sentiment(self, text: str, method: str = 'combined') -> float:
        """
        Analyze sentiment using specified method.
        
        Args:
            text: Text string to analyze
            method: Method to use ('textblob', 'vader', or 'combined')
            
        Returns:
            Sentiment score (-1 to 1)
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> score = analyzer.analyze_sentiment("Stock price target raised", method='vader')
        """
        if method == 'textblob':
            return self.analyze_sentiment_textblob(text)
        elif method == 'vader':
            return self.analyze_sentiment_vader(text)
        elif method == 'combined':
            # Average of both methods
            textblob_score = self.analyze_sentiment_textblob(text)
            vader_score = self.analyze_sentiment_vader(text)
            return (textblob_score + vader_score) / 2
        else:
            raise ValueError(f"Unknown method: {method}. Use 'textblob', 'vader', or 'combined'")
    
    def classify_sentiment(self, score: float, thresholds: Tuple[float, float] = (-0.1, 0.1)) -> str:
        """
        Classify sentiment score into category.
        
        Args:
            score: Sentiment score (-1 to 1)
            thresholds: Tuple of (negative_threshold, positive_threshold)
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> label = analyzer.classify_sentiment(0.5)
            >>> print(label)
            'positive'
        """
        if score < thresholds[0]:
            return 'negative'
        elif score > thresholds[1]:
            return 'positive'
        else:
            return 'neutral'
    
    def analyze_sentiment_batch(self, texts: pd.Series, method: str = 'combined') -> pd.Series:
        """
        Analyze sentiment for a series of texts.
        
        Args:
            texts: Series of text strings
            method: Method to use ('textblob', 'vader', or 'combined')
            
        Returns:
            Series of sentiment scores
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
        """
        return texts.apply(lambda x: self.analyze_sentiment(x, method=method))
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, date_col: str = 'date',
                                 sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Aggregate sentiment scores by date.
        
        Calculates mean, median, and count of sentiment scores per day.
        
        Args:
            df: DataFrame with date and sentiment columns
            date_col: Name of the date column
            sentiment_col: Name of the sentiment score column
            
        Returns:
            DataFrame with daily aggregated sentiment:
            - date: Trading date
            - mean_sentiment: Average sentiment score
            - median_sentiment: Median sentiment score
            - article_count: Number of articles
            - positive_count: Number of positive articles
            - negative_count: Number of negative articles
            - neutral_count: Number of neutral articles
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
            >>> daily = analyzer.aggregate_daily_sentiment(df)
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['date_only'] = df[date_col].dt.date
        
        # Classify sentiment
        df['sentiment_label'] = df[sentiment_col].apply(self.classify_sentiment)
        
        # Aggregate by date
        daily_stats = df.groupby('date_only').agg({
            sentiment_col: ['mean', 'median', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['date', 'mean_sentiment', 'median_sentiment', 'article_count', 'sentiment_distribution']
        
        # Extract sentiment counts
        daily_stats['positive_count'] = daily_stats['sentiment_distribution'].apply(
            lambda x: x.get('positive', 0) if isinstance(x, dict) else 0
        )
        daily_stats['negative_count'] = daily_stats['sentiment_distribution'].apply(
            lambda x: x.get('negative', 0) if isinstance(x, dict) else 0
        )
        daily_stats['neutral_count'] = daily_stats['sentiment_distribution'].apply(
            lambda x: x.get('neutral', 0) if isinstance(x, dict) else 0
        )
        
        # Drop distribution column
        daily_stats = daily_stats.drop(columns=['sentiment_distribution'])
        
        return daily_stats

