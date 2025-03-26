# LSTM Text Prediction Model Documentation

## Overview
This code implements a text prediction model using **LSTM networks** to predict the next word in a sequence. It includes data preprocessing, tokenization, model training/evaluation, and prediction visualization. The pipeline supports bidirectional and double-layer LSTM architectures.

---

## Features
- **Text Preprocessing**: Removes noise, stopwords, and punctuation.
- **Tokenization**: Converts text into integer sequences.
- **Model Architectures**:
  - Bidirectional LSTM
  - Double-Layer LSTM
- **Training Utilities**: Checkpointing, learning rate scheduling, and TensorBoard logging.
- **Prediction Analysis**: Generates CSV reports for model predictions.

---

## Workflow
1. **Data Loading**: Reads text from a file.
2. **Preprocessing**: Cleans and normalizes text.
3. **Tokenization**: Converts text to sequences.
4. **Sequence Generation**: Creates input-output pairs for training.
5. **Model Training**: Trains LSTM models with customizable hyperparameters.
6. **Evaluation**: Tests model accuracy and generates prediction reports.

---

## Functions

### `load_text_data(file_path)`
- **Purpose**: Load text data from a file.
- **Parameters**:
  - `filename` (str): Path to the input text file.
- **Returns**: Raw text as a string.

### `preprocess_text(text)`
- **Purpose**: Clean raw text.
- **Steps**:
  1. Remove tabs/newlines.
  2. Convert to lowercase.
  3. Remove numbers and special characters.
  4. Trim extra spaces.
- **Returns**: Cleaned text.

### `remove_stopwords(text)`
- **Purpose**: Filter out English stopwords using NLTK.
- **Returns**: Text with stopwords removed.

### `create_tokenizer(text, save_path)`
- **Purpose**: Tokenize text and save the tokenizer.
- **Parameters**:
  - `foldername` (str): Directory to save the tokenizer (`goethe_without_stopwords.pkl`).
- **Returns**:
  - `sequence_data` (list): Tokenized integer sequence.
  - `vocab_size` (int): Vocabulary size.

### `generate_sequences(sequence, seq_len, vocab_size, shuffle)`
- **Purpose**: Generate input-output sequences for training.
- **Parameters**:
  - `sequence_len` (int): Context window size (e.g., 10 words).
  - `visualization_status` (bool): Skips one-hot encoding if `True`.
- **Returns**:
  - `X` (numpy.ndarray): Input sequences.
  - `y` (numpy.ndarray): Target labels (one-hot encoded if `visualization_status=False`).

---

## Model Architectures
### Bidirectional LSTM
- Build bidirectional LSTM model.
- Architecture:
  - Embedding Layer
  - Bidirectional LSTM
  - Dense Output Layer
  ```python
  def build_bidirectional_lstm(nodes, embedding_size, sequence_len, vocab_size):
    model = Sequential([
          Embedding(vocab_size, embedding_dim, input_length=seq_length),
          Bidirectional(LSTM(lstm_units, return_sequences=False)),
          Dense(vocab_size, activation='softmax')
      ])
      
      model.compile(
          loss='categorical_crossentropy',
          optimizer=Adam(learning_rate=Config.LEARNING_RATE),
          metrics=['accuracy']
      )
      return model

### Double-Layer LSTM
- Build Double-Layer LSTM model.
- Architecture:
  - Embedding Layer
  - LSTM
  - LSTM
  - Dense Output Layer
  - Dense Output Layer
  ```python
  def build_model_double_lstm(nodes, embedding_size, sequence_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_len))
    model.add(LSTM(nodes, return_sequences=True))
    model.add(LSTM(nodes))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

##Training & Evaluation
### `train_model(model, X, y, epochs, batch_size, logdir, model_filename)`
- **Purpose**: Train the model with callbacks.
- **Callbacks**:
  - `ModelCheckpoint`:Saves the best model.
  - `ReduceLROnPlateau`: Reduces learning rate on plateau.
  - `TensorBoard`:Logs metrics for visualization.

- **Hyperparameters**:
  - `epochs`: Training iterations (default: 8-12).
  - `batch_size`: Samples per batch (default: 64).

### `test_model(X, y, logdir, model_filename)`
- **Purpose**: Evaluate model performance on test data.
- **Returns**: Test loss and accuracy.

### `visualization_of_model(X, y, model_filename, save_name)`
- **Purpose**: Generate CSV reports of predictions vs. actual values.
- **Outputs**:
  - `models_results_<save_name>.csv`: Full predictions.
  - `models_results_true_<save_name>.csv`: Correct predictions only.

## Usage
### Setup
1. **Install Dependencies**:
  - `pip install tensorflow nltk pandas matplotlib sklearn`
  - `python -m nltk.downloader stopwords punkt`

2. **Prepare Data**: Place text file (e.g., 1661-0.txt) in the working directory.

### Run Pipeline
- Execute function `train_pipline()` to start training and evaluation

### Hyperparameters (Customizable in trainings_pipeline())
  - `sequence_len`: Context window size (default: 10).
  - `embedding_size`: Embedding layer dimension (default: 100-500).
  - `nodes`: LSTM units per layer (default: 256).

## Outputs
- **Saved Models**: .h5 files (e.g., bidirectional_lstm_150_em500_8.h5).
- **Tokenizer**: goethe_without_stopwords.pkl.
- **CSV Reports**: Prediction results for analysis.
- **TensorBoard Logs**: Training metrics in logs_*/.

## Notes
- **Stopwords**: Removed by default for faster training (toggle via remove_stopwords()).
- **Hardcoded Paths**: Update filenames/directories (e.g., foldername, file) as needed.

- **GPU Acceleration**: Recommended for large datasets/hyperparameters.
