# Text Generation Using RNN, GRU, LSTM, and Transformer Models

## Introduction
This repository demonstrates the implementation of text generation models using four neural network architectures: RNN, GRU, LSTM, and Transformer. These models are trained on text data to generate sequences based on a given prompt. The pipeline includes data preprocessing, model training, and sequence generation, providing insights into the workings and performance of each model.

---

## Table of Contents
1. [Concepts and Theory](#concepts-and-theory)
    - [Text Generation Basics](#text-generation-basics)
    - [Mathematics Behind the Models](#mathematics-behind-the-models)
2. [Pipeline Overview](#pipeline-overview)
    - [Data Preprocessing](#data-preprocessing)
    - [Sequence Generation](#sequence-generation)
3. [Model Architectures](#model-architectures)
    - [RNN](#recurrent-neural-network-rnn)
    - [GRU](#gated-recurrent-unit-gru)
    - [LSTM](#long-short-term-memory-lstm)
    - [Transformer](#transformer)
4. [Usage Instructions](#usage-instructions)
5. [Results and Observations](#results-and-observations)
6. [Future Work](#future-work)

---

## Concepts and Theory

### Text Generation Basics
Text generation models aim to predict the next word in a sequence based on a given input. The process involves:
1. **Tokenization**: Splitting text into words or subwords.
2. **Sequence Encoding**: Mapping tokens to integers using a vocabulary.
3. **Model Training**: Learning patterns in the text to minimize prediction error.
4. **Generation**: Using the trained model to predict subsequent words.

### Mathematics Behind the Models
1. **Objective Function**:
   - Cross-Entropy Loss: Measures the divergence between predicted and true probability distributions.
     \[
     \mathcal{L}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
     \]
2. **Embedding Layer**:
   - Projects one-hot encoded vectors to dense representations.
     \[
     \mathbf{E} = \mathbf{W}_{embedding} \cdot \mathbf{x}
     \]
3. **Recurrent Computation**:
   - RNN: \( h_t = \sigma(\mathbf{W}_h h_{t-1} + \mathbf{W}_x x_t) \)
   - GRU: Combines update and reset gates to improve gradient flow.
   - LSTM: Adds a memory cell \( c_t \) to retain long-term dependencies.
   - Transformer: Uses self-attention \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \).

---

## Pipeline Overview

### Data Preprocessing
1. **Input Text**: Read from `my.txt`.
2. **Punctuation Removal**: Strip non-essential punctuation while retaining sentence-ending markers.
3. **Tokenization**: Convert text into lowercase tokens using NLTK.
4. **Encoding**:
   - Map tokens to integers using a dictionary (`word_to_int`).
   - Create sequences of fixed length (`seq_length`) with corresponding targets.

### Sequence Generation
For a given input prompt, the trained model predicts the next word iteratively by:
1. Tokenizing and encoding the input.
2. Feeding the encoded sequence to the model.
3. Decoding the predicted token back to text.

---

## Model Architectures

### Recurrent Neural Network (RNN)
- **Structure**:
  - Embedding Layer: Converts token indices to dense vectors.
  - RNN Layers: Two stacked RNN layers process the sequential data.
  - Fully Connected Layer: Maps the hidden state to output vocabulary size.
- **Equation**:
  \[ h_t = \tanh(\mathbf{W}_{hh} h_{t-1} + \mathbf{W}_{xh} x_t) \]

### Gated Recurrent Unit (GRU)
- **Enhancements**:
  - Update and Reset Gates improve gradient flow.
- **Equations**:
  \[ z_t = \sigma(\mathbf{W}_z x_t + \mathbf{U}_z h_{t-1}) \]
  \[ r_t = \sigma(\mathbf{W}_r x_t + \mathbf{U}_r h_{t-1}) \]
  \[ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(\mathbf{W}_h x_t + r_t \odot \mathbf{U}_h h_{t-1}) \]

### Long Short-Term Memory (LSTM)
- **Features**:
  - Memory Cell: Retains long-term dependencies.
  - Gates: Input, Forget, and Output gates control data flow.
- **Equations**:
  \[ f_t = \sigma(\mathbf{W}_f x_t + \mathbf{U}_f h_{t-1}) \]
  \[ c_t = f_t \odot c_{t-1} + i_t \odot \tanh(\mathbf{W}_c x_t + \mathbf{U}_c h_{t-1}) \]
  \[ h_t = o_t \odot \tanh(c_t) \]

### Transformer
- **Components**:
  - Positional Encoding: Adds sequence order information.
  - Multi-Head Self-Attention: Captures global dependencies.
  - Feedforward Network: Applies non-linear transformations.
- **Attention Equation**:
  \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

---

## Usage Instructions
1. Clone the repository.
2. Place the input text file (`my.txt`) in the root directory.
3. Install dependencies:
   ```bash
   pip install torch nltk numpy
   ```
4. Run the training scripts:
   - RNN: `python train_rnn.py`
   - GRU: `python train_gru.py`
   - LSTM: `python train_lstm.py`
   - Transformer: `python train_transformer.py`
5. Generate text:
   ```bash
   python generate_text.py --model <model_path> --start_words "<prompt>"
   ```

---

## Results and Observations
- **Training Loss**: All models achieved convergence with slight variations.
- **Text Quality**:
  - RNN: Struggled with long-term dependencies.
  - GRU: Improved coherence and flow.
  - LSTM: Generated the most coherent and contextually accurate text.
  - Transformer: Excelled in capturing global context and generated diverse outputs.

---

## Future Work
1. Experiment with larger datasets and longer sequences.
2. Implement pre-trained embeddings (e.g., GloVe, Word2Vec).
3. Fine-tune a Transformer-based model (e.g., GPT or BERT).
4. Incorporate beam search or nucleus sampling for text generation.

---

## Conclusion
This repository showcases the implementation and comparison of RNN, GRU, LSTM, and Transformer models for text generation. Each architecture has its strengths and weaknesses, with Transformers leading in capturing global dependencies and producing coherent text.

