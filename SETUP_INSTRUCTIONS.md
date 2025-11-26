# Setup Instructions

## Issue: ModuleNotFoundError for textblob

If you encounter `ModuleNotFoundError: No module named 'textblob'` in Jupyter notebooks, follow these steps:

## Solution

### 1. Activate Virtual Environment
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install All Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Jupyter Kernel (if using Jupyter)
```bash
python -m ipykernel install --user --name=news-sentiment-trading --display-name "Python (news-sentiment-trading)"
```

### 4. Restart Jupyter Notebook
- Close and restart your Jupyter notebook server
- Make sure to select the correct kernel: "Python (news-sentiment-trading)"

### 5. Verify Installation
Run this in a Python shell:
```python
import textblob
import vaderSentiment
import talib
import yfinance
print("All dependencies installed successfully!")
```

## Common Issues

### Issue: Jupyter uses different Python environment
**Solution**: Make sure Jupyter is running in the same virtual environment:
```bash
# Install jupyter in the venv
source venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=news-sentiment-trading
```

### Issue: TA-Lib installation fails
**Solution**: Install the C library first:
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

### Issue: textblob needs NLTK data
**Solution**: Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
```

## Quick Fix Command
Run this one-liner to install everything:
```bash
source venv/bin/activate && pip install -r requirements.txt && python -m ipykernel install --user --name=news-sentiment-trading --display-name "Python (news-sentiment-trading)" && echo "Setup complete! Restart Jupyter notebook."
```

