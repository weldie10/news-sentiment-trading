# Task 1: EDA Analysis Plan - Analyst Ratings Dataset

## Dataset Overview
- **Source**: `data/raw_analyst_ratings.csv`
- **Size**: ~1.4M records
- **Columns**: `headline`, `url`, `publisher`, `date`, `stock`
- **Domain**: Financial analyst ratings and news headlines

## Minimum Essential EDA Tasks

### 1. Descriptive Statistics

**Headline Length Analysis:**
```python
# Calculate headline character/word counts
df['headline_length'] = df['headline'].str.len()
df['headline_word_count'] = df['headline'].str.split().str.len()
# Generate: mean, median, std, min, max, quartiles
```

**Publisher Activity:**
```python
# Count articles per publisher
publisher_counts = df['publisher'].value_counts()
# Identify top 10-20 most active publishers
# Check for email-based publishers (contains '@')
```

**Publication Date Trends:**
```python
# Parse datetime, extract: year, month, day, hour, day_of_week
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
df['hour'] = df['date'].dt.hour
# Analyze: articles per day/month, peak hours, weekday patterns
```

### 2. Text Analysis (Topic Modeling)

**Keyword Extraction:**
```python
# Extract financial keywords using NLP
# Key phrases: "price target", "FDA approval", "earnings", "maintains", "raises", "lowers"
# Use NLTK/TextBlob for tokenization, remove stopwords
# Identify most frequent n-grams (2-3 words)
```

**Topic Modeling:**
```python
# Use scikit-learn LDA or NMF
# Extract topics: ratings changes, earnings, FDA approvals, market movements
# Visualize top keywords per topic
```

### 3. Time Series Analysis

**Publication Frequency:**
```python
# Daily article count time series
daily_counts = df.groupby(df['date'].dt.date).size()
# Identify spikes (market events, earnings seasons)
# Plot time series with annotations for major market events
```

**Publishing Time Patterns:**
```python
# Hourly distribution of publications
hourly_dist = df.groupby(df['date'].dt.hour).size()
# Identify peak trading hours (pre-market, market open, close)
# Critical for automated trading systems
```

### 4. Publisher Analysis

**Publisher Contribution:**
```python
# Top publishers by volume
# Publisher diversity (unique publishers count)
# Articles per publisher distribution
```

**Email Domain Analysis:**
```python
# Extract email domains from publisher column
email_publishers = df[df['publisher'].str.contains('@', na=False)]
email_publishers['domain'] = email_publishers['publisher'].str.split('@').str[1]
# Count by domain (e.g., benzinga.com contributors)
# Identify organizational patterns
```

**Publisher Content Differences:**
```python
# Analyze headline keywords by publisher
# Compare: individual authors vs. newsdesk vs. email contributors
# Identify specialized publishers (earnings, ratings, general news)
```

## Key Deliverables

1. **Statistics Summary**: Headline length stats, publisher counts, temporal distributions
2. **Visualizations**: 
   - Time series of publication frequency
   - Hourly distribution heatmap
   - Publisher activity bar chart
   - Top keywords word cloud
3. **Insights**: Peak publishing times, dominant publishers, common topics, market event correlations

## Industry Alignment

- **Trading Systems**: Identify optimal data refresh windows based on publishing patterns
- **Risk Management**: Understand news volume spikes during market volatility
- **Content Strategy**: Recognize which publishers drive most analyst rating news
- **Automation**: Time-based triggers for sentiment analysis pipelines

