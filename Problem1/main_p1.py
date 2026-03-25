
import os
import re
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import requests



from typing import List, Tuple, Optional, Dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from torch.utils.data import Dataset, DataLoader

"""#Text Preprocessing"""

#Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


#Extract text content from a PDF file.
def read_pdf_content(filepath):
    extracted_text = []

    try:
        with open(filepath, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)

        return "\n".join(extracted_text)

    except Exception as error:
        print(f"Error reading PDF '{filepath}': {error}")
        return ""

#Clean and normalize text by removing URLs, emails, special chars
def clean_document_text(text_content):
    #Convert to lowercase
    text_content = text_content.lower()

    #Removing URLs
    text_content = re.sub(r'https?://\S+|www\.\S+', '', text_content)

    #Remove email addresses
    text_content = re.sub(r'\S+@\S+\.\S+', '', text_content)

    #Keep only letters and spaces, remove everything else
    text_content = re.sub(r'[^a-z\s]', ' ', text_content)

    #Collapse multiple spaces and trim
    text_content = re.sub(r'\s+', ' ', text_content).strip()

    return text_content


#Processing all PDF files in a directory and combine their content
def process_pdf_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return ""

    combined_content = []
    pdf_files_found = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            full_path = os.path.join(directory_path, filename)
            print(f"Reading: {filename}")

            raw_content = read_pdf_content(full_path)
            if raw_content:
                cleaned = clean_document_text(raw_content)
                combined_content.append(cleaned)
                pdf_files_found += 1

    print(f"Processed {pdf_files_found} PDF files")
    return " ".join(combined_content)

def merge_corpus_files(output_filename="full_corpus.txt"):
    corpus_dir = "cleaned_corpus"
    merge_order = ["AcadReg.txt", "FacProfile.txt", "CurrNSyllabus.txt"]

    merged_content = []

    print(f"\nMerging individual corpus files...")

    for corpus_file in merge_order:
        full_path = os.path.join(corpus_dir, corpus_file)

        if not os.path.exists(full_path):
            print(f"Skipping missing file: {corpus_file}")
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                merged_content.append(content)
                print(f"Added {corpus_file}")

        except Exception as error:
            print(f"Failed to read {corpus_file}: {error}")

    if not merged_content:
        print("No valid content available to merge")
        return False, ""

    full_combined = " ".join(merged_content)
    output_path = os.path.join(corpus_dir, output_filename)

    save_text_content(full_combined, output_path)

    print(f"Created merged corpus: {len(full_combined.split()):,} total words")

    return True, full_combined

#Processing all .txt files in a directory and combine their content.
def process_text_directory(directory_path):

    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return ""

    combined_content = []
    text_files_found = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.txt'):
            full_path = os.path.join(directory_path, filename)
            print(f"Reading TXT: {filename}")

            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    combined_content.append(content)
                    text_files_found += 1
            except Exception as error:
                print(f"Could not read {filename}: {error}")

    if text_files_found:
        print(f"Processed {text_files_found} text files")
    else:
        print(f"No text files found in {directory_path}")

    return " ".join(combined_content)

#Saving text content to a file
def save_text_content(text, output_filepath):
    try:
        with open(output_filepath, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Saved: {output_filepath}")
        return True
    except Exception as error:
        print(f"Failed to save {output_filepath}: {error}")
        return False

#Ensuring cleaned corpus exists by processing source data
def prepare_corpus_files():
    corpus_dir = 'cleaned_corpus'
    source_dir = 'data'

    required_files = ['AcadReg.txt', 'FacProfile.txt', 'CurrNSyllabus.txt']

    #Create corpus directory if it doesn't exist
    Path(corpus_dir).mkdir(exist_ok=True)

    #Check if all required files already exist
    existing_files = all(
        os.path.exists(os.path.join(corpus_dir, f))
        for f in required_files
    )

    if existing_files:
        print("All corpus files already prepared")
        return True

    print("Some corpus files missing, processing source data...")

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found")
        return False

    #Define processing tasks for each document type
    processing_tasks = {
        'AcadReg': ('AcadReg', process_pdf_directory),
        'FacProfile': ('FacProfile', process_text_directory),
        'CurrNSyllabus': ('CurrNSyllabus', process_pdf_directory)
    }

    for file_key, (folder_name, processor) in processing_tasks.items():
        output_file = os.path.join(corpus_dir, f"{file_key}.txt")

        #Skip if file already exists
        if os.path.exists(output_file):
            print(f"{file_key}.txt already exists, skipping...")
            continue

        source_path = os.path.join(source_dir, folder_name)

        if not os.path.exists(source_path):
            print(f"Source folder missing: {source_path}")
            continue

        print(f"\nProcessing {folder_name}...")
        combined_text = processor(source_path)

        if combined_text:
            save_text_content(combined_text, output_file)
        else:
            print(f"No content extracted from {folder_name}")

    return True


#Calculating various statistics from text content
def calculate_text_statistics(text_content):
    #Additional cleaning for tokenization
    text_content = text_content.lower()
    text_content = re.sub(r'[^a-z\s]', '', text_content)
    text_content = re.sub(r'\s+', ' ', text_content).strip()

    #Tokenization and filtering
    tokens = word_tokenize(text_content)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    #Calculate statistics
    word_counts = Counter(filtered_tokens)
    total_words = len(filtered_tokens)
    unique_words = len(word_counts)

    return {
        'tokens': filtered_tokens,
        'word_counts': word_counts,
        'total_words': total_words,
        'unique_words': unique_words,
        'type_token_ratio': unique_words / total_words if total_words > 0 else 0
    }

#Generating and saving a word cloud visualization
def create_word_cloud(word_frequencies, output_path):
    if not word_frequencies:
        print("No data available for word cloud")
        return False

    cloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(word_frequencies)

    plt.figure(figsize=(12, 7))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Corpus', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    try:
        cloud.to_file(output_path)
        print(f"Word cloud saved to: {output_path}")
        return True
    except Exception as error:
        print(f"Failed to save word cloud: {error}")
        return False

#Analyzing individual document sections.
def analyze_document_section(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        stats = calculate_text_statistics(content)

        return {
            'name': os.path.basename(file_path),
            'total_tokens': stats['total_words'],
            'unique_words': stats['unique_words'],
            'top_words': stats['word_counts'].most_common(5)
        }

    except Exception as error:
        print(f"Error analyzing {file_path}: {error}")
        return None
#Main function
def main():

    corpus_file = "Data/full_corpus.txt"
    corpus_dir = "Data"

    #Step 1: Check if full corpus already exists
    if not os.path.exists(corpus_file):
        print("1:- full_corpus.txt not found. Preparing document corpus...")

        if not prepare_corpus_files():
            print("Failed to prepare corpus. Exiting.")
            return

        merge_ok, full_merged_text = merge_corpus_files()
        if not merge_ok:
            print("Merging failed. Exiting.")
            return

        full_corpus = full_merged_text

    else:
        print("1:- full_corpus.txt already exists. Loading and skipping preparation...")

        with open(corpus_file, "r", encoding="utf-8") as f:
            full_corpus = f.read()

    #Step 2 & 3 ONLY if needed for per-file analysis
    text_files = []
    if os.path.exists(corpus_dir):
        text_files = [
            os.path.join(corpus_dir, f)
            for f in os.listdir(corpus_dir)
            if f.endswith('.txt') and f != "full_corpus.txt"
        ]

    #Analyze corpus
    corpus_stats = calculate_text_statistics(full_corpus)

    print("\n4:-Corpus Statistics:")
    print("-" * 40)
    print(f"Total Tokens: {corpus_stats['total_words']:,}")
    print(f"Vocabulary Size: {corpus_stats['unique_words']:,}")

    if text_files:
        print(f"Total Documents: {len(text_files)}")
        avg_tokens = corpus_stats['total_words'] / len(text_files)
        print(f"Average Tokens per Document: {avg_tokens:.0f}")

    print(f"Type-Token Ratio: {corpus_stats['type_token_ratio']:.4f}")

    #Visualization
    print("\n5:-Creating visualizations...")
    wordcloud_path = os.path.join(corpus_dir, 'wordcloud.png')
    create_word_cloud(corpus_stats['word_counts'], wordcloud_path)

    #Individual section analysis
    if not os.path.exists(corpus_file):
      do_section_analysis = True
    else:
        do_section_analysis = False
    if do_section_analysis:
        print("\n6:-Individual Section Analysis:")
        print("-" * 40)

        for file_path in text_files:
            section_stats = analyze_document_section(file_path)
            if section_stats:
                print(f"\n({section_stats['name']}):")
                print(f"  Tokens: {section_stats['total_tokens']:,}")
                print(f"  Unique Words: {section_stats['unique_words']:,}")

                top_five = section_stats['top_words']
                if top_five:
                    word_list = ", ".join([f"{w}({c})" for w, c in top_five])
                    print(f"  Top Words: {word_list}")

main()

"""#Model"""

#Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



############################################## Data Loading N Preprocessing ##############################################

#Reading and tokenizing the corpus file
def load_corpus(filepath: str) -> List[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        #Basic tokenization split on whitespace , keep meaningful words
        tokens = [word for word in text.split() if len(word) >= 2]
        print(f"Loaded {len(tokens):,} tokens")
        return tokens

    except FileNotFoundError:
        print(f"Error: Couldn't find {filepath}")
        return []

#Creating vocabulary from tokens with frequency filtering
def build_vocab(tokens: List[str], min_freq: int = 2) -> Tuple[dict, dict, int]:
    word_counts = Counter(tokens)

    #Filter rare words
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    vocab.sort()  #Keep consistent ordering

    #Create mapping dictionaries
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    print(f"Vocabulary: {len(vocab):,} words (min freq={min_freq})")
    return word_to_idx, idx_to_word, len(vocab)

#Removing tokens not in vocab
def filter_tokens(tokens: List[str], vocab_dict: dict) -> List[str]:

    filtered = [t for t in tokens if t in vocab_dict]
    removed = len(tokens) - len(filtered)
    if removed:
        print(f"Filtered out {removed:,} unknown tokens")
    return filtered


def subsample_tokens(tokens: List[str], word_counts: Counter, threshold: float = 1e-3) -> List[str]:
    #Reduce frequency of very common words using subsampling
    total = len(tokens)
    kept = []

    for word in tokens:
        freq = word_counts.get(word, 0) / total
        if freq == 0:
            continue

        #Probability to keep this word
        keep_prob = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
        keep_prob = min(keep_prob, 1.0)

        if random.random() < keep_prob:
            kept.append(word)

    print(f"After subsampling: {len(kept):,} tokens ({len(kept)/len(tokens)*100:.1f}% kept)")
    return kept


#Create probability distribution for negative sampling
def create_negative_distribution(vocab_size: int, idx_to_word: dict,
                                word_counts: Counter, power: float = 0.75) -> np.ndarray:

    counts = np.array([word_counts.get(idx_to_word[i], 0) for i in range(vocab_size)])
    scaled = np.power(counts, power)
    return scaled / scaled.sum()


def split_train_val(tokens: List[str], val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    """Spliting tokens into training and validation sets."""
    split_point = int(len(tokens) * (1 - val_ratio))
    train = tokens[:split_point]
    val = tokens[split_point:]

    print(f"Training: {len(train):,} tokens | Validation: {len(val):,} tokens")
    return train, val



############################################## Dataset Classes ##############################################

#Dataset for CBOW model , context --> target word
class CBOWDataset(Dataset):

    def __init__(self, tokens: List[str], word_to_idx: dict, window: int):
        self.samples = []

        for center in range(window, len(tokens) - window):
            target = tokens[center]
            if target not in word_to_idx:
                continue

            context = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                context_word = tokens[center + offset]
                if context_word in word_to_idx:
                    context.append(word_to_idx[context_word])

            if context:
                self.samples.append((context, word_to_idx[target]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram model , center -> context word"""

    def __init__(self, tokens: List[str], word_to_idx: dict, window: int):
        self.samples = []

        for center in range(window, len(tokens) - window):
            center_word = tokens[center]
            if center_word not in word_to_idx:
                continue

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                context_word = tokens[center + offset]
                if context_word in word_to_idx:
                    self.samples.append((word_to_idx[center_word],
                                        word_to_idx[context_word]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)



############################################## Model Definitions ##############################################

#Base class for word embedding models
class BaseEmbeddingModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.input_embeds = nn.Embedding(vocab_size, embed_dim)
        self.output_embeds = nn.Embedding(vocab_size, embed_dim)

        self._init_weights()

    #Initializing embedding weights
    def _init_weights(self):

        nn.init.uniform_(self.input_embeds.weight, -0.5/self.embed_dim, 0.5/self.embed_dim)
        nn.init.zeros_(self.output_embeds.weight)

    def get_word_vector(self, word: str, word_to_idx: dict) -> Optional[np.ndarray]:
        """Get embedding vector for a single word"""
        if word not in word_to_idx:
            return None
        idx = torch.tensor(word_to_idx[word], dtype=torch.long).to(DEVICE)
        return self.input_embeds(idx).detach().cpu().numpy()

     #Finding most similar words using cosine similarity
    def find_similar(self, query_word: str, word_to_idx: dict,
                    idx_to_word: dict, top_k: int = 5) -> List[Tuple[str, float]]:

        if query_word not in word_to_idx:
            print(f"Word '{query_word}' not in vocab.")
            return []

        #get all embeddings
        embeddings = self.input_embeds.weight.detach().cpu().numpy()
        query_idx = word_to_idx[query_word]
        query_vec = embeddings[query_idx]

        #Normalizing for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeds = embeddings / (norms + 1e-8)

        query_norm = np.linalg.norm(query_vec)
        query_vec_norm = query_vec / (query_norm + 1e-8)

        #Computing similarities
        similarities = norm_embeds @ query_vec_norm
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            word = idx_to_word[idx]
            if word != query_word:
                results.append((word, float(similarities[idx])))
                if len(results) >= top_k:
                    break
        return results

    #Saving model weights
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Saved model to {path}")

    def load(self, path: str) -> bool:
        """Load model weights"""
        try:
            state_dict = torch.load(path, map_location=DEVICE)

            #Check if shapes match current model
            current = {k: v.shape for k, v in self.state_dict().items()}
            for key, tensor in state_dict.items():
                if key in current and tensor.shape != current[key]:
                    print(f"Shape mismatch for {key}: saved {tensor.shape}, current {current[key]}")
                    return False

            self.load_state_dict(state_dict)
            self.to(DEVICE)
            return True

        except FileNotFoundError:
            print(f"Model file not found: {path}")
            return False


#continuous Bag of Words model
class CBOWModel(BaseEmbeddingModel):

    def forward(self, context, target, negatives):
        #Average context embeddings
        context_vecs = self.input_embeds(context)  #[batch, context_len, dim]
        hidden = context_vecs.mean(dim=1)          #[batch, dim]

        #Positive score
        target_vecs = self.output_embeds(target)   #[batch, dim]
        pos_score = torch.sum(hidden * target_vecs, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        #Negative scores
        neg_vecs = self.output_embeds(negatives)   #[batch, neg_count, dim]
        hidden_expanded = hidden.unsqueeze(2)      #[batch, dim, 1]
        neg_scores = torch.bmm(neg_vecs, hidden_expanded).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_scores) + 1e-10)

        #Total loss
        return -(pos_loss + neg_loss.sum(dim=1)).mean()


class SkipGramModel(BaseEmbeddingModel):
    """Skip-gram model."""

    def forward(self, center, context, negatives):
        #Center word embedding
        center_vecs = self.input_embeds(center)   #[batch, dim]

        #Positive score
        context_vecs = self.output_embeds(context)  #[batch, dim]
        pos_score = torch.sum(center_vecs * context_vecs, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        #Negative scores
        neg_vecs = self.output_embeds(negatives)    #[batch, neg_count, dim]
        center_expanded = center_vecs.unsqueeze(2)  #[batch, dim, 1]
        neg_scores = torch.bmm(neg_vecs, center_expanded).squeeze(2)
        neg_loss = torch.log(torch.sigmoid(-neg_scores) + 1e-10)

        #Total loss
        return -(pos_loss + neg_loss.sum(dim=1)).mean()



############################################## Training Utilities ##############################################

def collate_cbow(batch):
    """Collate function for CBOW variable context length"""
    contexts, targets = zip(*batch)
    max_len = max(len(c) for c in contexts)

    padded = []
    for ctx in contexts:
        pad_needed = max_len - len(ctx)
        padded.append(torch.cat([ctx, torch.zeros(pad_needed, dtype=torch.long)]))

    return torch.stack(padded), torch.stack(list(targets))

#Sample negative words for a batch
def sample_negatives(batch_size: int, neg_count: int, dist: np.ndarray) -> torch.Tensor:
    indices = np.random.choice(len(dist), size=(batch_size, neg_count), p=dist)
    return torch.tensor(indices, dtype=torch.long).to(DEVICE)


#Train a model for multiple epochs
def train_epochs(model: nn.Module, dataset: Dataset, negative_dist: np.ndarray,
                epochs: int = 5, batch_size: int = 512, lr: float = 0.01,
                neg_samples: int = 5, is_cbow: bool = True):

    print(f"Training on {len(dataset):,} examples")
    model.to(DEVICE)

    #Set up data loader
    if is_cbow:
        loader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_cbow)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0

        for batch in loader:
            actual_batch = batch[0].size(0)
            negatives = sample_negatives(actual_batch, neg_samples, negative_dist)

            if is_cbow:
                context, target = batch
                context, target = context.to(DEVICE), target.to(DEVICE)
                loss = model(context, target, negatives)
            else:
                center, context = batch
                center, context = center.to(DEVICE), context.to(DEVICE)
                loss = model(center, context, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


#Evaluate model on validation set
def eval_model(model: nn.Module, dataset: Dataset, negative_dist: np.ndarray,
              neg_samples: int = 5, batch_size: int = 512,
              is_cbow: bool = True) -> float:

    model.eval()

    if is_cbow:
        loader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_cbow)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            actual_batch = batch[0].size(0)
            negatives = sample_negatives(actual_batch, neg_samples, negative_dist)

            if is_cbow:
                context, target = batch
                context, target = context.to(DEVICE), target.to(DEVICE)
                loss = model(context, target, negatives)
            else:
                center, context = batch
                center, context = center.to(DEVICE), context.to(DEVICE)
                loss = model(center, context, negatives)

            total_loss += loss.item()
            batches += 1

    model.train()
    return total_loss / max(batches, 1)


def pick_best_param(results: Dict, param_name: str):
    """Select best parameter based on validation loss"""
    best_val = min(results, key=lambda k: results[k]["val_loss"])
    best_loss = results[best_val]["val_loss"]

    print(f"\n{'─' * 40}")
    print(f"  {param_name} Comparison:")
    for param, info in sorted(results.items(), key=lambda x: x[1]["val_loss"]):
        mark = "  <- BEST" if param == best_val else ""
        print(f"    {param_name}={param:<4}  val_loss={info['val_loss']:.4f}{mark}")
    print(f"  Selected {param_name} = {best_val} (loss={best_loss:.4f})")
    print(f"{'─' * 40}")

    return best_val


#Train new model or load existing one
def train_or_reuse(model_class, name: str, vocab_size: int, dim: int,
                  window: int, dataset: Dataset, negative_dist: np.ndarray,
                  epochs: int = 5, neg_samples: int = 5,
                  is_cbow: bool = True) -> nn.Module:

    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pt"
    model = model_class(vocab_size, dim)

    if model.load(path):
        print(f"Loaded existing model: {name}")
        return model

    print(f"Training new model: {name}")
    train_epochs(model, dataset, negative_dist, epochs=epochs,
                neg_samples=neg_samples, is_cbow=is_cbow)
    model.save(path)
    return model



############################################## Evaluation Utilities ##############################################

#Computing cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return dot / norm


#Solving word analogy: word_a is to word_b as word_c is to ?
def analogy(model, word_a: str, word_b: str, word_c: str,
            word_to_idx: dict, idx_to_word: dict,
            top_n: int = 5) -> List[Tuple[str, float]]:

    for w in [word_a, word_b, word_c]:
        if w not in word_to_idx:
            print(f"  '{w}' not in vocabulary")
            return []

    #Get all embedding weights
    all_embeds = model.input_embeds.weight.detach().cpu().numpy()

    vec_a = all_embeds[word_to_idx[word_a]]
    vec_b = all_embeds[word_to_idx[word_b]]
    vec_c = all_embeds[word_to_idx[word_c]]

    #Compute analogy vector: b - a + c
    target_vec = vec_b - vec_a + vec_c

    #Score all words except the three input words
    exclude = {word_a, word_b, word_c}
    scores = []

    for i in range(len(idx_to_word)):
        word = idx_to_word[i]
        if word in exclude:
            continue
        sim = cosine_similarity(target_vec, all_embeds[i])
        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]



############################################## Visualization Helpers ##############################################

#Semantic word clusters for visualization
WORD_CLUSTERS = {
    "academic degrees": ["btech", "mtech", "phd", "undergraduate", "postgraduate", "diploma"],
    "research":         ["research", "thesis", "publication", "project", "lab", "experiment"],
    "people":           ["student", "faculty", "professor", "researcher", "advisor", "scholar"],
    "assessment":       ["exam", "grade", "marks", "evaluation", "assignment", "result"],
    "departments":      ["mathematics", "physics", "chemistry", "computer", "electrical", "mechanical"],
}

#Color mapping for each cluster
CLUSTER_COLORS = {
    "academic degrees": "#e74c3c",
    "research":         "#3498db",
    "people":           "#2ecc71",
    "assessment":       "#f39c12",
    "departments":      "#9b59b6",
}


#Extracting embeddings for words that exist in vocabulary
def get_cluster_embeddings(model, clusters: Dict[str, List[str]],
                           word_to_idx: dict) -> Tuple[List[str], np.ndarray, List[str]]:

    words = []
    vectors = []
    labels = []

    for cluster_name, cluster_words in clusters.items():
        for word in cluster_words:
            if word in word_to_idx:
                words.append(word)
                vec = model.input_embeds.weight[word_to_idx[word]].detach().cpu().numpy()
                vectors.append(vec)
                labels.append(cluster_name)
            else:
                print(f"  Skipping '{word}' - not in vocabulary")

    return words, np.array(vectors), labels


#Plotting reduced embeddings with cluster coloring
def plot_embeddings(reduced: np.ndarray, words: List[str], labels: List[str],
                    title: str, filename: str):

    plt.figure(figsize=(14, 9))

    #Plotting each cluster separately for legend
    unique_labels = list(dict.fromkeys(labels))
    for cluster in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == cluster]
        x = [reduced[i, 0] for i in idxs]
        y = [reduced[i, 1] for i in idxs]

        plt.scatter(x, y, c=CLUSTER_COLORS[cluster], label=cluster,
                    s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

        #Annotating each point with the word
        for i, idx in enumerate(idxs):
            plt.annotate(words[idx], (x[i], y[i]),
                         fontsize=8, alpha=0.9,
                         xytext=(5, 5), textcoords="offset points")

    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="best", fontsize=9, framealpha=0.7)
    plt.xlabel("Component 1", fontsize=10)
    plt.ylabel("Component 2", fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {filename}")



############################################## Main Pipeline ##############################################

if __name__ == "__main__":

    ############ Step 1: Data Preparation ############

    print("\n1:- Loading corpus...")
    tokens = load_corpus("cleaned_corpus/full_corpus.txt")
    if not tokens:
        exit(1)

    #Build vocabulary
    word_to_idx, idx_to_word, vocab_size = build_vocab(tokens, min_freq=2)

    #Filter tokens
    filtered = filter_tokens(tokens, word_to_idx)

    #Word counts for subsampling
    counts = Counter(filtered)

    #Apply subsampling
    print("\n2:- Applying subsampling")
    sampled = subsample_tokens(filtered, counts)

    #Split into train/validation
    print("\n3:- Creating train/validation split")
    train_tokens, val_tokens = split_train_val(sampled, val_ratio=0.1)

    #Create negative sampling distribution
    print("\n4:- Setting up negative sampling")
    train_counts = Counter(train_tokens)
    negative_dist = create_negative_distribution(vocab_size, idx_to_word, train_counts)


    ############ Step 2: Hyperparameter Experiments ############

    #Experimenting with different embedding dimensions for CBOW
    print("\n" + "=" * 60)
    print("Experiment 1a;- Finding Best Embedding Dimension (CBOW)")
    print("=" * 60)

    dims = [50, 100, 200]
    cbow_dim_results = {}

    for dim in dims:
        print(f"\n  Testing CBOW with dim={dim}...")

        train_ds = CBOWDataset(train_tokens, word_to_idx, window=5)
        val_ds = CBOWDataset(val_tokens, word_to_idx, window=5)

        model = train_or_reuse(CBOWModel, f"cbow_dim{dim}", vocab_size, dim,
                              5, train_ds, negative_dist)

        val_loss = eval_model(model, val_ds, negative_dist, is_cbow=True)
        print(f"  Validation loss: {val_loss:.4f}")
        cbow_dim_results[dim] = {"val_loss": val_loss}

    best_cbow_dim = pick_best_param(cbow_dim_results, "dim")

    #Experimenting with different embedding dimensions for Skip-gram
    print("\n" + "=" * 60)
    print("Experiment 1b;- Finding Best Embedding Dimension (Skip-gram)")
    print("=" * 60)

    sg_dim_results = {}

    for dim in dims:
        print(f"\n  Testing Skip-gram with dim={dim}...")

        train_ds = SkipGramDataset(train_tokens, word_to_idx, window=5)
        val_ds = SkipGramDataset(val_tokens, word_to_idx, window=5)

        model = train_or_reuse(SkipGramModel, f"sg_dim{dim}", vocab_size, dim,
                              5, train_ds, negative_dist, is_cbow=False)

        val_loss = eval_model(model, val_ds, negative_dist, is_cbow=False)
        print(f"  Validation loss: {val_loss:.4f}")
        sg_dim_results[dim] = {"val_loss": val_loss}

    best_sg_dim = pick_best_param(sg_dim_results, "dim")

    #Experimenting with different window sizes for CBOW
    print("\n" + "=" * 60)
    print("Experiment 2a: Finding Best Window Size (CBOW)")
    print("=" * 60)

    windows = [2, 5, 10]
    cbow_window_results = {}

    for window in windows:
        print(f"\n  Testing CBOW with window={window} (dim={best_cbow_dim})...")

        train_ds = CBOWDataset(train_tokens, word_to_idx, window=window)
        val_ds = CBOWDataset(val_tokens, word_to_idx, window=window)

        model = train_or_reuse(CBOWModel, f"cbow_win{window}", vocab_size,
                              best_cbow_dim, window, train_ds, negative_dist)

        val_loss = eval_model(model, val_ds, negative_dist, is_cbow=True)
        print(f"  Validation loss: {val_loss:.4f}")
        cbow_window_results[window] = {"val_loss": val_loss}

    best_cbow_window = pick_best_param(cbow_window_results, "window")

    #Experimenting with different window sizes for Skip-gram
    print("\n" + "=" * 60)
    print("Experiment 2b: Finding Best Window Size (Skip-gram)")
    print("=" * 60)

    sg_window_results = {}

    for window in windows:
        print(f"\n  Testing Skip-gram with window={window} (dim={best_sg_dim})...")

        train_ds = SkipGramDataset(train_tokens, word_to_idx, window=window)
        val_ds = SkipGramDataset(val_tokens, word_to_idx, window=window)

        model = train_or_reuse(SkipGramModel, f"sg_win{window}", vocab_size,
                              best_sg_dim, window, train_ds, negative_dist, is_cbow=False)

        val_loss = eval_model(model, val_ds, negative_dist, is_cbow=False)
        print(f"  Validation loss: {val_loss:.4f}")
        sg_window_results[window] = {"val_loss": val_loss}

    best_sg_window = pick_best_param(sg_window_results, "window")

    #Experimenting with different negative sample counts for CBOW
    print("\n" + "=" * 60)
    print("Experiment 3a: Finding Best Negative Samples (CBOW)")
    print("=" * 60)

    neg_counts = [3, 5, 10]
    cbow_neg_results = {}

    for neg in neg_counts:
        print(f"\n  Testing CBOW with neg_samples={neg} (dim={best_cbow_dim}, window={best_cbow_window})...")

        train_ds = CBOWDataset(train_tokens, word_to_idx, window=best_cbow_window)
        val_ds = CBOWDataset(val_tokens, word_to_idx, window=best_cbow_window)

        model = train_or_reuse(CBOWModel, f"cbow_neg{neg}", vocab_size, best_cbow_dim,
                              best_cbow_window, train_ds, negative_dist, neg_samples=neg)

        val_loss = eval_model(model, val_ds, negative_dist, neg_samples=neg, is_cbow=True)
        print(f"  Validation loss: {val_loss:.4f}")
        cbow_neg_results[neg] = {"val_loss": val_loss}

    best_cbow_neg = pick_best_param(cbow_neg_results, "neg_samples")

    #Experimenting with different negative sample counts for Skip-gram
    print("\n" + "=" * 60)
    print("Experiment 3b: Finding Best Negative Samples (Skip-gram)")
    print("=" * 60)

    sg_neg_results = {}

    for neg in neg_counts:
        print(f"\n  Testing Skip-gram with neg_samples={neg} (dim={best_sg_dim}, window={best_sg_window})...")

        train_ds = SkipGramDataset(train_tokens, word_to_idx, window=best_sg_window)
        val_ds = SkipGramDataset(val_tokens, word_to_idx, window=best_sg_window)

        model = train_or_reuse(SkipGramModel, f"sg_neg{neg}", vocab_size, best_sg_dim,
                              best_sg_window, train_ds, negative_dist,
                              neg_samples=neg, is_cbow=False)

        val_loss = eval_model(model, val_ds, negative_dist, neg_samples=neg, is_cbow=False)
        print(f"  Validation loss: {val_loss:.4f}")
        sg_neg_results[neg] = {"val_loss": val_loss}

    best_sg_neg = pick_best_param(sg_neg_results, "neg_samples")


    ############ Step 3: Final Model Training ############

    print("\n" + "=" * 60)
    print("Final Model Training")
    print(f"  CBOW using: dim={best_cbow_dim}, window={best_cbow_window}, neg_samples={best_cbow_neg}")
    print(f"  Skip-gram using: dim={best_sg_dim}, window={best_sg_window}, neg_samples={best_sg_neg}")
    print("=" * 60)

    print("\nTraining final CBOW model...")
    cbow_train = CBOWDataset(train_tokens, word_to_idx, window=best_cbow_window)
    cbow_val = CBOWDataset(val_tokens, word_to_idx, window=best_cbow_window)

    final_cbow = train_or_reuse(CBOWModel, "final_cbow", vocab_size, best_cbow_dim,
                               best_cbow_window, cbow_train, negative_dist,
                               epochs=10, neg_samples=best_cbow_neg)
    cbow_loss = eval_model(final_cbow, cbow_val, negative_dist,
                          neg_samples=best_cbow_neg, is_cbow=True)
    print(f"  Final CBOW validation loss: {cbow_loss:.4f}")

    print("\nTraining final Skip-gram model...")
    sg_train = SkipGramDataset(train_tokens, word_to_idx, window=best_sg_window)
    sg_val = SkipGramDataset(val_tokens, word_to_idx, window=best_sg_window)

    final_sg = train_or_reuse(SkipGramModel, "final_skipgram", vocab_size, best_sg_dim,
                             best_sg_window, sg_train, negative_dist,
                             epochs=10, neg_samples=best_sg_neg, is_cbow=False)
    sg_loss = eval_model(final_sg, sg_val, negative_dist,
                        neg_samples=best_sg_neg, is_cbow=False)
    print(f"  Final Skip-gram validation loss: {sg_loss:.4f}")

    #Training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  CBOW best dimension:       {best_cbow_dim}")
    print(f"  CBOW best window:          {best_cbow_window}")
    print(f"  CBOW best neg_samples:     {best_cbow_neg}")
    print(f"  CBOW loss:                 {cbow_loss:.4f}")
    print(f"  Skip-gram best dimension:  {best_sg_dim}")
    print(f"  Skip-gram best window:     {best_sg_window}")
    print(f"  Skip-gram best neg_samples:{best_sg_neg}")
    print(f"  Skip-gram loss:            {sg_loss:.4f}")


    ############ Step 4: Word Similarity Test ############

    print("\n" + "=" * 60)
    print("Word Similarity Test")
    print("=" * 60)

    test_words = ["research", "student", "phd", "exam", "college"]

    print("\nCBOW Model:")
    for word in test_words:
        similar = final_cbow.find_similar(word, word_to_idx, idx_to_word)
        if similar:
            print(f"\n  Words similar to '{word}':")
            for sim_word, score in similar:
                print(f"    {sim_word:<20} {score:.4f}")

    print("\n" + "-" * 40)
    print("\nSkip-gram Model:")
    for word in test_words:
        similar = final_sg.find_similar(word, word_to_idx, idx_to_word)
        if similar:
            print(f"\n  Words similar to '{word}':")
            for sim_word, score in similar:
                print(f"    {sim_word:<20} {score:.4f}")


    ############ Step 5: Nearest Neighbors ############

    print("\n" + "=" * 60)
    print("Task 3.1: Nearest Neighbors")
    print("=" * 60)

    models_to_test = {
        "CBOW":      final_cbow,
        "Skip-gram": final_sg,
    }

    query_words = ["research", "student", "phd", "exam"]

    for name, model in models_to_test.items():
        print(f"\n{'─' * 40}")
        print(f"  {name} Model")
        print(f"{'─' * 40}")

        for word in query_words:
            results = model.find_similar(word, word_to_idx, idx_to_word, top_k=5)
            if results:
                print(f"\n  Words similar to '{word}':")
                print(f"  {'Rank':<6} {'Word':<20} {'Similarity':>10}")
                print(f"  {'-' * 38}")
                for rank, (sim_word, score) in enumerate(results, 1):
                    print(f"  {rank:<6} {sim_word:<20} {score:>10.4f}")
            else:
                print(f"\n  '{word}' not found in vocabulary")


    ############ Step 6: Analogy Experiments ############

    print("\n" + "=" * 60)
    print("Task 3.2: Analogy Experiments")
    print("=" * 60)

    #Format: (word_a, word_b, word_c, expected_answer)
    analogy_tests = [
        ("ug",       "btech",         "pg",        "mtech"),
        ("student",  "exam",          "faculty",   "evaluation"),
        ("btech",    "undergraduate", "mtech",     "postgraduate"),
        ("research", "phd",           "teaching",  "professor"),
        ("theory",   "mathematics",   "experiment", "physics"),
    ]

    for name, model in models_to_test.items():
        print(f"\n{'─' * 40}")
        print(f"  {name} Model")
        print(f"{'─' * 40}")

        for word_a, word_b, word_c, expected in analogy_tests:
            print(f"\n  {word_a} : {word_b} :: {word_c} : ?")
            print(f"  (expected: '{expected}')")

            results = analogy(model, word_a, word_b, word_c,
                            word_to_idx, idx_to_word, top_n=5)
            if results:
                print(f"  Top predictions:")
                for rank, (word, score) in enumerate(results, 1):
                    match = "  <-- correct" if word == expected else ""
                    print(f"    {rank}. {word:<20} {score:.4f}{match}")
            else:
                print("  Could not compute, missing words in vocabulary")


    ############ Step 7: Visualization ############

    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    models_to_viz = {
        "CBOW":      final_cbow,
        "Skip-gram": final_sg,
    }

    #Generating PCA and t-SNE plots for each model
    for name, model in models_to_viz.items():
        print(f"\n{'─' * 40}")
        print(f"  Visualizing: {name}")
        print(f"{'─' * 40}")

        words, vectors, labels = get_cluster_embeddings(model, WORD_CLUSTERS, word_to_idx)
        print(f"  Found {len(words)} words in vocabulary")

        if len(words) < 5:
            print("  Too few words to visualize, check word clusters")
            continue

        #PCA reduction
        print("\n  Running PCA...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(vectors)
        var_explained = pca.explained_variance_ratio_

        plot_embeddings(
            pca_result, words, labels,
            title=f"{name} - PCA Projection\n"
                  f"(variance: PC1={var_explained[0]:.2%}, PC2={var_explained[1]:.2%})",
            filename=f"viz_{name.lower().replace('-', '')}_pca.png"
        )

        #t-SNE reduction
        print("\n  Running t-SNE...")
        perplexity = min(30, len(words) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity,
                     random_state=42, max_iter=1000)
        tsne_result = tsne.fit_transform(vectors)

        plot_embeddings(
            tsne_result, words, labels,
            title=f"{name} - t-SNE Projection (perplexity={perplexity})",
            filename=f"viz_{name.lower().replace('-', '')}_tsne.png"
        )

    #Side-by-side comparison of both models using PCA
    print("\n" + "=" * 60)
    print("Side-by-Side Comparison (PCA)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (name, model) in zip(axes, models_to_viz.items()):
        words, vectors, labels = get_cluster_embeddings(model, WORD_CLUSTERS, word_to_idx)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        #Plotting clusters on this subplot
        unique_labels = list(dict.fromkeys(labels))
        for cluster in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == cluster]
            x = [reduced[i, 0] for i in idxs]
            y = [reduced[i, 1] for i in idxs]

            ax.scatter(x, y, c=CLUSTER_COLORS[cluster], label=cluster,
                       s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

            for i, idx in enumerate(idxs):
                ax.annotate(words[idx], (x[i], y[i]),
                            fontsize=8, alpha=0.9,
                            xytext=(5, 5), textcoords="offset points")

        ax.set_title(f"{name} - PCA", fontsize=13)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle("CBOW vs Skip-gram: Word Embedding Clusters (PCA)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("viz_comparison_pca.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: viz_comparison_pca.png")

    print("\n" + "=" * 60)
    print("All tasks complete!!")
    print("=" * 60)

print("\n" + "=" * 60)
print("Side-by-Side Comparison (t-SNE)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, (name, model) in zip(axes, models_to_viz.items()):
    words, vectors, labels = get_cluster_embeddings(model, WORD_CLUSTERS, word_to_idx)

    perplexity = min(30, len(words) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                 random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(vectors)

    #Plotting clusters on this subplot
    unique_labels = list(dict.fromkeys(labels))
    for cluster in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == cluster]
        x = [reduced[i, 0] for i in idxs]
        y = [reduced[i, 1] for i in idxs]

        ax.scatter(x, y, c=CLUSTER_COLORS[cluster], label=cluster,
                   s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

        for i, idx in enumerate(idxs):
            ax.annotate(words[idx], (x[i], y[i]),
                        fontsize=8, alpha=0.9,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_title(f"{name} - t-SNE (perplexity={perplexity})", fontsize=13)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

plt.suptitle("CBOW vs Skip-gram: Word Embedding Clusters (t-SNE)",
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("viz_comparison_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: viz_comparison_tsne.png")
