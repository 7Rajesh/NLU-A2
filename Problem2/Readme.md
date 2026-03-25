# NLU Assignment 2 — P2: Character-Level Name Generation

Trains and compares three neural models to generate new names character by character:

- **Vanilla RNN** — baseline recurrent network
- **Bidirectional LSTM** — processes sequences in both directions
- **RNN with Attention** — uses attention over past characters for better context

Models are evaluated on **novelty rate** (names not seen in training) and **diversity** (unique names generated) across temperatures `0.6`, `0.8`, and `1.0`.

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install torch numpy pandas
   ```

2. **Add dataset** — place `names.txt` (one name per line) in the directory  `Data`

3. **Run**
   ```bash
   python main_p2.py
   ```

Generated names are saved to `Output/generated_names.txt` after evaluation.
