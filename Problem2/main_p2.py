import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#DATA LOADING AND PREPROCESSING

#Load names from file each line contains one name set(name.strip().lower() for name in training_names)
with open("Data/names.txt", "r", encoding="utf-8") as f:
    all_names = [line.strip().lower() for line in f if line.strip()]

# Get unique names and limit to 1000
names = list(set(all_names))[:1000]

print(f"Loaded {len(names)} unique names from dataset (max 1000)")

#Build vocabulary from all unique characters in the dataset
#We need special tokens for padding, start of sequence, and end of sequence
chars = sorted(set("".join(names)))
chars = ["<pad>", "<sos>", "<eos>"] + chars

#Create character to index and index to character mappings
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

#Store important constants
vocab_size = len(chars)
PAD = c2i["<pad>"]  #Padding token for batching sequences of different lengths
SOS = c2i["<sos>"]  #Start of sequence token
EOS = c2i["<eos>"]  #End of sequence token

print(f"Vocabulary size: {vocab_size} characters")



###############DATASET AND DATA LOADING ###############

class NamesDataset(Dataset):
    """
    Custom dataset for name sequences
    Each name is converted to a sequence of indices with SOS and EOS tokens
    """
    def __init__(self, names):
        self.data = []
        for name in names:
            #Convert: "abc" -> [SOS, a, b, c, EOS]
            seq = [SOS] + [c2i[c] for c in name] + [EOS]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Pad sequences in a batch to the same length
    """
    max_len = max(len(x) for x in batch)
    padded = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), PAD, dtype=torch.long)])
        for x in batch
    ])
    return padded


#Create dataset and dataloader
dataset = NamesDataset(names)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)



###############VANILLA RNN ###############


class VanillaRNN(nn.Module):
    """
    Simple RNN model for character-level language modeling.r
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Embedding layer: converts character indices to dense vectors
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)

        #RNN layer: processes sequences while maintaining hidden state
        self.rnn = nn.RNN(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,  #Input shape: (batch, sequence, features)
            dropout=0.2 if num_layers > 1 else 0  #Dropout between RNN layers
        )

        #Dropout for regularization before final layer
        self.dropout = nn.Dropout(0.2)

        #Output layer: maps hidden state to vocabulary probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):

        #Convert indices to embeddings
        embedded = self.embed(x)

        #Process through RNN
        rnn_out, h = self.rnn(embedded, h)

        #Apply dropout and get predictions
        output = self.fc(self.dropout(rnn_out))

        return output, h

    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def num_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    #Generate a new name character by character
    def generate(self, max_len=15, temperature=0.8):



        self.eval()  #Set model to evaluation mode

        with torch.no_grad():  #No gradient computation needed
            #Start with SOS token
            x = torch.tensor([[SOS]]).to(device)
            h = self.init_hidden(1)
            name = []

            for _ in range(max_len):
                #Get predictions for next character
                logits, h = self.forward(x, h)

                #Apply temperature to logits and convert to probabilities
                #Lower temperature makes the model more confident (less random)
                #Higher temperature makes the model more diverse (more random)
                probs = torch.softmax(logits[0, -1] / temperature, dim=0)

                #Sample next character from probability distribution
                c = torch.multinomial(probs, 1).item()

                #Stop if we generate EOS token
                if c == EOS:
                    break

                #Add character to name (skip special tokens)
                if c not in [PAD, SOS]:
                    name.append(i2c[c])

                #Use this character as input for next step
                x = torch.tensor([[c]]).to(device)

        return "".join(name) if name else "<empty>"



###############BIDIRECTIONAL LSTM ###############

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM model
    Processes sequences in both forward and backward directions
    This gives the model more context but makes generation trickier
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)

        #Bidirectional LSTM: processes sequences forward and backward
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,  #Key difference from vanilla RNN
            dropout=0.3 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(0.3)

        #Output size is doubled because we concatenate forward and backward
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embed(x)

        #LSTM processes in both directions
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)

        #lstm_out contains concatenated forward and backward outputs
        output = self.fc(self.dropout(lstm_out))

        return output, hidden

    def init_hidden(self, batch_size):

        h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h, c)

    def num_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, max_len=15, temperature=0.8):

        self.eval()

        with torch.no_grad():
            #Start with SOS token
            current_input = torch.tensor([[SOS]]).to(device)
            generated_seq = [SOS]
            name = []

            #Initialize hidden state
            h, c = self.init_hidden(1)

            for _ in range(max_len):
                #Forward pass through the model
                embedded = self.embed(current_input)
                lstm_out, (h, c) = self.lstm(embedded, (h, c))

                #Get logits from the last timestep
                logits = self.fc(lstm_out[:, -1, :])

                #Sample next character using temperature
                probs = torch.softmax(logits[0] / temperature, dim=0)
                next_char = torch.multinomial(probs, 1).item()

                #Check for end of sequence
                if next_char == EOS:
                    break

                #Add to name if not a special token
                if next_char not in [PAD, SOS]:
                    name.append(i2c[next_char])
                    generated_seq.append(next_char)

                #Prepare input for next step
                current_input = torch.tensor([[next_char]]).to(device)

        return "".join(name) if name else "<empty>"



###############RNN WITH ATTENTION MECHANISM ###############

#Attention mechanism that learns to focus on different parts of the sequence
class Attention(nn.Module):



    def __init__(self, hidden_size):
        super().__init__()
        #Transform hidden states before computing attention scores
        self.attn = nn.Linear(hidden_size, hidden_size)
        #Convert to scalar attention scores
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_out):

        #Compute attention scores for each position
        scores = self.v(torch.tanh(self.attn(rnn_out)))

        #Convert scores to probabilities (weights sum to 1)
        weights = torch.softmax(scores, dim=1)

        #Compute weighted sum of RNN outputs
        context = torch.sum(weights * rnn_out, dim=1)

        return context, weights


class RNNAttention(nn.Module):
    """
    RNN with attention mechanism
    Combines RNN outputs with attention-weighted context for predictions
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #Standard components
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        #Attention mechanism
        self.attention = Attention(hidden_size)

        self.dropout = nn.Dropout(0.2)

        #Output layer takes both RNN output and attention context
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h=None):

        #Get embeddings
        embedded = self.embed(x)

        #Process through RNN
        rnn_out, h = self.rnn(embedded, h)

        #Compute attention over all RNN outputs
        context, _ = self.attention(rnn_out)

        #Expand context to match sequence length and concatenate with RNN output
        #This gives each position access to the global context
        context_expanded = context.unsqueeze(1).expand(-1, rnn_out.size(1), -1)
        combined = torch.cat([rnn_out, context_expanded], dim=2)

        #Generate predictions
        output = self.fc(self.dropout(combined))

        return output, h

    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def num_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, max_len=15, temperature=0.8):

        self.eval()

        with torch.no_grad():
            #Store all RNN outputs for attention computation
            all_outputs = []
            h = self.init_hidden(1)

            #Start with SOS token
            current_input = torch.tensor([[SOS]]).to(device)
            name = []

            for _ in range(max_len):
                #Process current character through RNN
                embedded = self.embed(current_input)
                rnn_out, h = self.rnn(embedded, h)

                #Store this output for attention
                all_outputs.append(rnn_out)

                #Compute attention over all previous outputs
                if len(all_outputs) > 1:
                    #We have multiple outputs to attend over
                    combined_rnn_out = torch.cat(all_outputs, dim=1)
                    context, _ = self.attention(combined_rnn_out)
                else:
                    #First character: no previous context
                    context = rnn_out.squeeze(1)

                #Combine current RNN output with attention context
                current_rnn_out = rnn_out[:, -1, :]
                combined = torch.cat([current_rnn_out, context], dim=1)

                #Get predictions
                logits = self.fc(combined)

                #Sample next character
                probs = torch.softmax(logits[0] / temperature, dim=0)
                next_char = torch.multinomial(probs, 1).item()

                #Check for end of sequence
                if next_char == EOS:
                    break

                #Add to name if not special token
                if next_char not in [PAD, SOS]:
                    name.append(i2c[next_char])

                #Prepare next input
                current_input = torch.tensor([[next_char]]).to(device)

        return "".join(name) if name else "<empty>"



###############TRAINING FUNCTION ###############

def train(model, loader, epochs=20, lr=0.001, stop_loss=0.8, model_name="model"):

    model.to(device)

    #Loss function: ignore padding tokens when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    #Optimizer: Adam works well for RNNs
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Learning rate scheduler: reduce LR if loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5
    )

    #Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch in loader:
            batch = batch.to(device)

            #Split into input and target sequences
            #Input: everything except last character
            #Target: everything except first character (SOS)
            inp, tgt = batch[:, :-1], batch[:, 1:]

            #Initialize hidden state
            if hasattr(model, 'init_hidden'):
                h = model.init_hidden(batch.size(0))
                #Detach to prevent backprop through time across batches
                if isinstance(h, tuple):
                    h = tuple(x.detach() for x in h)
                else:
                    h = h.detach()
            else:
                h = None

            #Forward pass
            if h is None:
                logits, _ = model(inp)
            else:
                logits, _ = model(inp, h)

            #Calculate loss
            #Reshape to (batch * seq_len, vocab_size) for loss computation
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1)
            )

            #Backward pass
            optimizer.zero_grad()
            loss.backward()

            #Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        #Calculate average loss for this epoch
        avg_loss = total_loss / batch_count

        #Update learning rate based on loss
        scheduler.step(avg_loss)

        print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")

        #Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            #Save the best model
            torch.save(model.state_dict(), f"models2/{model_name}_best.pt")
        else:
            patience_counter += 1
            #Early stopping if no improvement
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement)")
                break

        #Stop if we've reached target loss
        if avg_loss < stop_loss:
            print(f"  Target loss reached at epoch {epoch+1}")
            break

    #Load the best model weights
    model.load_state_dict(
        torch.load(f"models2/{model_name}_best.pt", map_location=device)
    )

    return model


###############UTILITY FUNCTIONS FOR SAVING,LOADING ###############
def save(model, path):
    """Save model weights to disk"""
    torch.save(model.state_dict(), path)
    print(f"Model saved: {path}")


def load(model, path):
    """
    Load model weights from disk.
    Returns True if successful, False if file not found.
    """
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded: {path}")
        return True
    except FileNotFoundError:
        return False



################MAIN TRAINING ###############

#Create directory for saving models
os.makedirs("models2", exist_ok=True)

os.makedirs("Output", exist_ok=True)


#Hyperparameters - these control the model architecture and training
EMBED = 32      #Size of character embeddings
HIDDEN = 64     #Size of hidden state in RNN/LSTM
LAYERS = 1      #Number of stacked RNN/LSTM layers
EPOCHS = 30     #Maximum training epochs
LR = 0.001      #Learning rate


#Train Model 1: Vanilla RNN

print("\n" + "="*50)
print("Training Vanilla RNN")
print("="*50)

rnn = VanillaRNN(vocab_size, EMBED, HIDDEN, LAYERS)
print(f"Parameters: {rnn.num_params():,}")

#Load existing model or train new one
if not load(rnn, "models2/rnn.pt"):
    rnn = train(rnn, loader, EPOCHS, LR, stop_loss=1.0, model_name="rnn")
    save(rnn, "models2/rnn.pt")


#Train Model 2: Bidirectional LSTM

print("\n" + "="*50)
print("Training Bidirectional LSTM")
print("="*50)

blstm = BiLSTM(vocab_size, EMBED, HIDDEN, LAYERS)
print(f"Parameters: {blstm.num_params():,}")

if not load(blstm, "models2/blstm.pt"):
    blstm = train(blstm, loader, EPOCHS, LR, stop_loss=1.0, model_name="blstm")
    save(blstm, "models2/blstm.pt")


#Train Model 3: RNN with Attention

print("\n" + "="*50)
print("Training RNN with Attention")
print("="*50)

attn_rnn = RNNAttention(vocab_size, EMBED, HIDDEN, LAYERS)
print(f"Parameters: {attn_rnn.num_params():,}")

if not load(attn_rnn, "models2/rnn_attn.pt"):
    attn_rnn = train(attn_rnn, loader, EPOCHS, LR, stop_loss=1.0, model_name="rnn_attn")
    save(attn_rnn, "models2/rnn_attn.pt")



#MODEL COMPARISON AND EVALUATION


print("\n" + "="*55)
print("MODEL SUMMARY")
print("="*55)
print(f"{'Model':<22} {'Params':>10} {'Hidden':>8} {'Layers':>8} {'Embed':>8}")
print("-" * 55)
print(f"{'Vanilla RNN':<22} {rnn.num_params():>10,} {HIDDEN:>8} {LAYERS:>8} {EMBED:>8}")
print(f"{'Bidirectional LSTM':<22} {blstm.num_params():>10,} {HIDDEN:>8} {LAYERS:>8} {EMBED:>8}")
print(f"{'RNN + Attention':<22} {attn_rnn.num_params():>10,} {HIDDEN:>8} {LAYERS:>8} {EMBED:>8}")

print("\n" + "="*50)
print("GENERATED NAME SAMPLES")
print("="*50)

#Test generation with different temperatures
#Temperature controls randomness:
#  - Low (0.6): More predictable, conservative
#  - Medium (0.8): Balanced
#  - High (1.0): More creative, diverse
temperatures = [0.6, 0.8, 1.0]

for temp in temperatures:
    print(f"\nTemperature: {temp}")
    print("-" * 30)

    #Generate 5 samples from each model
    rnn_samples = [rnn.generate(temperature=temp) for _ in range(5)]
    blstm_samples = [blstm.generate(temperature=temp) for _ in range(5)]
    attn_samples = [attn_rnn.generate(temperature=temp) for _ in range(5)]

    print(f"RNN:          {rnn_samples}")
    print(f"BiLSTM:       {blstm_samples}")
    print(f"RNN+Attention: {attn_samples}")


#ADDITIONAL DIAGNOSTICS


print("\n" + "="*50)
print("ADDITIONAL DIAGNOSTICS")
print("="*50)

#Check diversity: how many unique names can each model generate?
print("\nChecking model diversity (10 samples at temperature 1.0):")
for model_name, model in [("RNN", rnn), ("BiLSTM", blstm), ("RNN+Attn", attn_rnn)]:
    samples = set([model.generate(temperature=1.0) for _ in range(10)])
    print(f"{model_name:<12}: {len(samples)} unique out of 10")

###############EVALUATION METRICS###############

def evaluate_models(models_dict, training_names, n_samples=200, temperatures=[0.6, 0.8, 1.0]):
    """
    Evaluate novelty rate and diversity for each model.
    Novelty Rate: % of generated names NOT in training set
    Diversity:    unique names / total generated names
    """

    #Normalize training names for fair comparison
    #lowercase and strip to match generation output format
    training_set = set(name.strip().lower() for name in training_names)

    print(f"Training set size: {len(training_set)} unique names")
    print(f"Generating {n_samples} samples per model per temperature\n")

    #Store all results for comparison
    #Structure: results[model_name][temperature] = {metric: value}
    results = {}

    for model_name, model in models_dict.items():
        print(f"Evaluating: {model_name}")
        print("-" * 40)

        results[model_name] = {}

        for temp in temperatures:

            #Generate n_samples names
            generated = []
            for _ in range(n_samples):
                name = model.generate(temperature=temp)
                generated.append(name)

            #Filter out empty generations
            valid = [n for n in generated if n != "<empty>"]
            empty_count = n_samples - len(valid)

            #METRIC 1: NOVELTY RATE
            #Count names that do NOT appear in training set
            novel_names    = [n for n in valid if n not in training_set]
            novelty_rate   = len(novel_names) / len(valid) * 100 if valid else 0

            #METRIC 2: DIVERSITY
            #Unique names divided by total generated names
            unique_names   = set(valid)
            diversity      = len(unique_names) / len(valid) * 100 if valid else 0

            #Store results
            results[model_name][temp] = {
                "total_generated"   : n_samples,
                "valid"             : len(valid),
                "empty"             : empty_count,
                "novel_count"       : len(novel_names),
                "novelty_rate"      : novelty_rate,
                "unique_count"      : len(unique_names),
                "diversity"         : diversity,
                "generated_sample"  : valid[:8],       #store 8 samples
                "novel_sample"      : novel_names[:5], #store 5 novel examples
            }

            print(f"  Temperature {temp}:")
            print(f"    Valid generations : {len(valid)}/{n_samples}")
            print(f"    Empty generations : {empty_count}")
            print(f"    Novelty Rate      : {novelty_rate:.1f}%  "
                  f"({len(novel_names)}/{len(valid)} names not in training set)")
            print(f"    Diversity         : {diversity:.1f}%  "
                  f"({len(unique_names)} unique out of {len(valid)})")
            print(f"    Sample generated  : {valid[:5]}")
            print(f"    Sample novel      : {novel_names[:5]}")
            print()

        print()

    return results


def print_comparison_table(results, temperatures):

    #NOVELTY RATE TABLE
    print("=" * 75)
    print("NOVELTY RATE COMPARISON  (% of generated names NOT in training set)")
    print("=" * 75)

    #Header
    header = f"{'Model':<22}"
    for temp in temperatures:
        header += f"  Temp={temp}"
    print(header)
    print("-" * 75)

    #One row per model
    for model_name, temp_results in results.items():
        row = f"{model_name:<22}"
        for temp in temperatures:
            val = temp_results[temp]["novelty_rate"]
            row += f"  {val:>7.1f}%"
        print(row)

    print()

    #DIVERSITY TABLE
    print("=" * 75)
    print("DIVERSITY COMPARISON  (% unique names out of total generated)")
    print("=" * 75)

    header = f"{'Model':<22}"
    for temp in temperatures:
        header += f"  Temp={temp}"
    print(header)
    print("-" * 75)

    for model_name, temp_results in results.items():
        row = f"{model_name:<22}"
        for temp in temperatures:
            val = temp_results[temp]["diversity"]
            row += f"  {val:>7.1f}%"
        print(row)

    print()

    #COMBINED SCORE TABLE
    #Average of novelty and diversity gives a single quality score
    print("=" * 75)
    print("COMBINED SCORE  (average of Novelty Rate and Diversity)")
    print("=" * 75)

    header = f"{'Model':<22}"
    for temp in temperatures:
        header += f"  Temp={temp}"
    print(header)
    print("-" * 75)

    for model_name, temp_results in results.items():
        row = f"{model_name:<22}"
        for temp in temperatures:
            nov = temp_results[temp]["novelty_rate"]
            div = temp_results[temp]["diversity"]
            combined = (nov + div) / 2
            row += f"  {combined:>7.1f}%"
        print(row)

    print()


def print_best_model(results, temperatures):

    print("=" * 75)
    print("BEST MODEL PER METRIC AND TEMPERATURE")
    print("=" * 75)

    for temp in temperatures:
        print(f"\nTemperature = {temp}:")
        print(f"  {'Metric':<20} {'Best Model':<25} {'Score':>8}")
        print(f"  {'-'*55}")

        #Best novelty
        best_nov_model = max(
            results.keys(),
            key=lambda m: results[m][temp]["novelty_rate"]
        )
        best_nov_val = results[best_nov_model][temp]["novelty_rate"]
        print(f"  {'Novelty Rate':<20} {best_nov_model:<25} {best_nov_val:>7.1f}%")

        #Best diversity
        best_div_model = max(
            results.keys(),
            key=lambda m: results[m][temp]["diversity"]
        )
        best_div_val = results[best_div_model][temp]["diversity"]
        print(f"  {'Diversity':<20} {best_div_model:<25} {best_div_val:>7.1f}%")

        #Best combined
        best_comb_model = max(
            results.keys(),
            key=lambda m: (results[m][temp]["novelty_rate"] +
                          results[m][temp]["diversity"]) / 2
        )
        best_comb_nov = results[best_comb_model][temp]["novelty_rate"]
        best_comb_div = results[best_comb_model][temp]["diversity"]
        best_comb_val = (best_comb_nov + best_comb_div) / 2
        print(f"  {'Combined Score':<20} {best_comb_model:<25} {best_comb_val:>7.1f}%")


def print_overall_winner(results, temperatures):

    print("\n" + "=" * 75)
    print("OVERALL WINNER (averaged across all temperatures)")
    print("=" * 75)

    model_scores = {}

    for model_name, temp_results in results.items():
        avg_novelty   = sum(temp_results[t]["novelty_rate"] for t in temperatures) / len(temperatures)
        avg_diversity = sum(temp_results[t]["diversity"]    for t in temperatures) / len(temperatures)
        avg_combined  = (avg_novelty + avg_diversity) / 2

        model_scores[model_name] = {
            "avg_novelty"  : avg_novelty,
            "avg_diversity": avg_diversity,
            "avg_combined" : avg_combined
        }

    print(f"\n{'Model':<22} {'Avg Novelty':>13} {'Avg Diversity':>15} {'Avg Combined':>14}")
    print("-" * 68)

    for model_name, scores in model_scores.items():
        print(f"{model_name:<22} "
              f"{scores['avg_novelty']:>12.1f}% "
              f"{scores['avg_diversity']:>14.1f}% "
              f"{scores['avg_combined']:>13.1f}%")

    #Find overall winner
    winner = max(model_scores.keys(), key=lambda m: model_scores[m]["avg_combined"])
    print(f"\n  Overall Winner: {winner}  "
          f"(Combined Score: {model_scores[winner]['avg_combined']:.1f}%)")


###############RUN EVALUATION###############

#Pack models into a dictionary for clean iteration
models_dict = {
    "Vanilla RNN"   : rnn,
    "BiLSTM"        : blstm,
    "RNN+Attention" : attn_rnn
}

temperatures = [0.6, 0.8, 1.0]

print("\n" + "=" * 75)
print("RUNNING EVALUATION")
print("=" * 75 + "\n")

#Run evaluation
results = evaluate_models(
    models_dict    = models_dict,
    training_names = names,        #original loaded names list
    n_samples      = 200,          #generate 200 names per model per temperature
    temperatures   = temperatures
)

#Print comparison tables
print_comparison_table(results, temperatures)

#Print best model per metric
print_best_model(results, temperatures)

output_filename = "Output/generated_names.txt"

with open(output_filename, "w", encoding="utf-8") as f:

    # Iterate through each model
    for model_name, model in models_dict.items():
        f.write(f"Model: {model_name}\n")
        # f.write("-" * (len(model_name) + 7) + "\n")
            # Generate 10 names for each model at each temperature
        generated_names = [model.generate(temperature=0.9) for _ in range(10)]
        for i, name in enumerate(generated_names):
            f.write(f"    {i+1}. {name}\n")
        f.write("\n")
    f.write("\n")

print(f"Generated names saved to {output_filename}")

