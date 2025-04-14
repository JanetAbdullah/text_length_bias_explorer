# Text Length Bias Explorer

This project investigates whether there is a correlation or bias between the **length of a text** and its **sentiment label** (positive or negative). Using a fully synthetic dataset, we simulate reviews and perform statistical, visual, and predictive analyses to understand the potential influence of text length on sentiment classification.

## Project Objectives
- Generate a balanced synthetic dataset with varying text lengths
- Explore distribution of text lengths by sentiment
- Apply statistical testing (T-test) and correlation analysis
- Train a logistic regression model to predict sentiment from length and word-level features
- Visualize results with histograms, boxplots, scatter plots, heatmaps, and word clouds
- Evaluate fairness via residual analysis

## Tools Used
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- wordcloud

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels wordcloud
python text_length_bias_explorer.py
```

## File Structure
```
text_length_bias_explorer/
├── text_length_bias_explorer.py   # Full script with analysis
├── README.md                      # Project overview and usage
```

## Analysis Breakdown
###  Dataset Generation
- Synthetic reviews (1200 total)
- Length distributions: longer for positive, shorter for negative
- Average word length calculated

### Exploratory Analysis
- Histogram of length by sentiment
- Boxplot comparison
- Summary statistics by sentiment label

### Statistical Testing
- T-test to compare means of length between classes
- Pearson and Spearman correlation between text length and sentiment

### Logistic Regression
- Features: length and avg_word_len
- Model trained using `statsmodels.Logit`
- Residuals analyzed to evaluate prediction fairness

###  Visualization
- Predicted probability of sentiment by length
- Heatmap of correlations
- Word clouds for positive and negative texts

## Key Insights
- Positive texts are statistically longer than negative ones
- There is a weak but significant correlation between text length and sentiment
- Logistic regression confirms both text length and word richness influence classification
- Residuals centered around zero suggest limited bias from the model

## Author
Janet Abdullah  
GitHub: [https://github.com/JanetAbdullah]  
Feel free to explore, fork, and adapt this project for your portfolio.
