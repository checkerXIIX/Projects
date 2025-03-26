from pathlib import Path
from typing import Tuple, Generator
import re
import string
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# Configuration
class Config:
    SEQUENCE_LENGTH = 10
    EMBEDDING_DIM = 256
    LSTM_UNITS = 256
    BATCH_SIZE = 128
    EPOCHS = 15
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 0.001

def load_hf_dataset(dataset_name: str = "wikitext", 
                   config_name: str = "wikitext-103-v1") -> str:
    """Load and concatenate a Hugging Face dataset.
    
    Args:
        dataset_name: Name of dataset from HF Hub
        config_name: Dataset configuration/version
        
    Returns:
        Combined text from the dataset
    """
    dataset = load_dataset(dataset_name, config_name)
    
    # Combine all splits (train/validation/test)
    full_text = ""
    for split in dataset.keys():
        for example in dataset[split]:
            full_text += example["text"] + "\n"
    
    return full_text


def load_text_data(file_path: str) -> str:
    """Load text data from a file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Raw text content as string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_text(text: str) -> str:
    """Clean and normalize text data.
    
    Processing steps:
    1. Remove newlines and special characters
    2. Convert to lowercase
    3. Remove numbers and punctuation
    4. Remove single characters
    5. Remove extra whitespace
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    # Remove special characters and normalize whitespace
    text = re.sub(r'\ufeff|[“”„]', '', text)
    text = text.lower().replace('\n', ' ')
    
    # Remove numbers and punctuation
    text = re.sub(r'[0-9]+', '', text)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # Clean remaining artifacts
    text = re.sub(r'\b\w\b', '', text)  # Single characters
    text = re.sub(r'\s+', ' ', text)     # Multiple whitespace
    
    return text.strip()

def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text.
    
    Args:
        text: Preprocessed text
        
    Returns:
        Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

def create_tokenizer(text: str, save_path: str) -> Tuple[Tokenizer, np.ndarray, int]:
    """Create and save text tokenizer.
    
    Args:
        text: Processed text corpus
        save_path: Path to save tokenizer
        
    Returns:
        Tuple containing:
        - Fitted tokenizer
        - Tokenized sequence
        - Vocabulary size
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    # Save tokenizer
    with open(save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle)
        
    sequence = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1
    
    return tokenizer, np.array(sequence), vocab_size

def generate_sequences(
    sequence: np.ndarray, 
    seq_length: int, 
    vocab_size: int, 
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training sequences and targets.
    
    Args:
        sequence: Array of token indices
        seq_length: Context window size
        vocab_size: Vocabulary size
        shuffle: Whether to shuffle sequences
        
    Returns:
        Tuple of (padded_sequences, one_hot_targets)
    """
    sequences = []
    for i in range(seq_length, len(sequence)):
        sequences.append(sequence[i-seq_length:i+1])
    
    sequences = np.array(sequences)
    if shuffle:
        np.random.shuffle(sequences)
    
    X = sequences[:, :-1]
    y = sequences[:, -1]
    
    return X, to_categorical(y, num_classes=vocab_size)


def build_model_double_lstm(nodes, embedding_size, sequence_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_len))
    model.add(LSTM(nodes, return_sequences=True))
    model.add(LSTM(nodes))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def build_bidirectional_lstm(
    vocab_size: int,
    seq_length: int = Config.SEQUENCE_LENGTH,
    embedding_dim: int = Config.EMBEDDING_DIM,
    lstm_units: int = Config.LSTM_UNITS
) -> Sequential:
    """Build bidirectional LSTM model.
    
    Architecture:
    - Embedding Layer
    - Bidirectional LSTM
    - Dense Output Layer
    
    Args:
        vocab_size: Size of vocabulary
        seq_length: Input sequence length
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
        
    Returns:
        Compiled Keras model
    """
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

# Modified training pipeline
def train_pipeline_hf(
    dataset_name: str = "wikitext",
    config_name: str = "wikitext-103-v1",
    model_save_path: str = "hf_lstm_model.h5"
) -> None:
    """Training pipeline with Hugging Face integration."""
    # Load dataset
    raw_text = load_hf_dataset(dataset_name, config_name)
    
    # Rest of the pipeline remains the same
    processed_text = preprocess_text(raw_text)
    filtered_text = remove_stopwords(processed_text)
    
    # Tokenization
    tokenizer, sequence, vocab_size = create_tokenizer(filtered_text, "hf_tokenizer.pkl")
    
    # Train-test split
    split_idx = int(len(sequence) * (1 - test_size))
    train_seq, test_seq = sequence[:split_idx], sequence[split_idx:]
    
    # Sequence generation
    X_train, y_train = generate_sequences(train_seq, Config.SEQUENCE_LENGTH, vocab_size)
    X_test, y_test = generate_sequences(test_seq, Config.SEQUENCE_LENGTH, vocab_size, shuffle=False)
    
    # Model setup
    model = build_bidirectional_lstm(vocab_size)
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        TensorBoard(log_dir='logs')
    ]
    
    # Training
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=Config.VALIDATION_SPLIT,
        callbacks=callbacks
    )
    
    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.2%} | Test loss: {test_loss:.4f}')


if __name__ == '__main__':
    train_pipeline_hf(
        dataset_name="wikitext",
        config_name="wikitext-103-v1",
        model_save_path="wiki_lstm_model.h5"
    )
