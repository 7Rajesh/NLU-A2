# NLU Assignment 2 —P1: CBOW and Skip-gram Word Embeddings Based on Data Available on IITJ Website

Trains and compares two Word2Vec models — CBOW and Skip-gram — on text scraped from the IIT Jodhpur website. Embeddings are evaluated through validation loss, nearest neighbour analysis, analogy tests, and PCA/t-SNE visualizations.

---

## How to Run  

1. **Install dependencies**
   ```bash
   pip install torch numpy matplotlib nltk scikit-learn wordcloud requests PyPDF2
   ```

2. **Add corpus** — place `corpus.txt` (cleaned IIT Jodhpur text) in the `Data/` folder, or run the scraper to generate it automatically

3. **Run**
   ```bash
   python main_p1.py
   ```

Outputs saved: `Output/wordcloud.png`, `Output/viz_comparison_pca.png`, `Output/viz_comparison_tsne.png`, trained model files
