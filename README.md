# Chatbot

## Description

A simple chatbot built using a from scratch implementation of the Transformer architecture it demonstrates the construction of a sequence to sequence model for chat responses.
trained on a dataset of conversation pairs.

## Features
        - Transformer model implemented component wise
        - training on extracted dataset from pdf.
        - CUDA support for GPU acceleration.

## Requirements
        - Python 3.8+
        - PyTorch 1.10+
        - Other dependencies: numpy, torch (install via `pip install -r requirements.txt`)

## Installation
        1. Clone the repository:
        ```
        git clone https://github.com/IshtishadAlamTishad/Chatbot.git
        cd Chatbot
        ```
        2. Install dependencies (assuming standard PyTorch setup):
        ```
        pip install torch numpy
        ```
        3. Prepare data: Place your custom data in `asset/data/data.txt` (format: question\tanswer per line).

## Usage
### Training the Model
        Run the Jupyter notebook for training:
        ```
        jupyter notebook model.ipynb
        ```
        This will preprocess data, train the Transformer model, and save it to `model/trainedModel.pth`.

### Running the Chatbot
        Use the run script for inference:
        
        ```
        python run.py
        ```

Interact with the bot in the console by typing messages.


## Project Structure
        The project is organized as follows:
        - **asset/data/**: Raw data files (e.g., `data.txt` containing conversation pairs).
        - **asset/preprocessed/**: Processed datasets (e.g., `dataset.txt` and `extractedData.json` for tokenized or structured data).
        - **model/**: Saved model weights (e.g., `trainedModel.pth`).
        - **module/**: Core Python modules implementing the Transformer components.

        Chatbot:
        │
        │   LICENSE
        │   model.ipynb
        │   README.md
        │   requirements.txt
        │   run.py
        │
        ├───asset
        │   ├───data
        │   │       data.txt
        │   │
        │   └───preprocessed
        │           dataset.txt
        │           extractedData.json
        │
        ├───model
        │       trainedModel.pth
        │
        └───module
        │   bot.py
        │   createData.py
        │   cuda.py
        │   decoder.py
        │   dl.py
        │   el.py
        │   encoder.py
        │   LN.py
        │   multiheadAttention.py
        │   multiheadCattention.py
        │   pff.py
        │   postionalEncoding.py
        │   readData.py
        │   scaledDotProduct.py
        │   se.py
        │   seqDec.py
        │   sequencialEncoding.py
        │   sqEnc.py
        │   TestModel.py
        │   TrainModel.py
        │   Transformer.py
        │
        ├───VectorDB
        │   │   VD.py
        │   │
        │   └───__pycache__
        │           cache files
        │
        └───__pycache__
                cache files

## File Descriptions
Below is a description of each code file in the `module/` directory and `model.ipynb`, inferred from standard Transformer implementations and file names

### model.ipynb
        This Jupyter notebook likely handles the end-to-end workflow for model training. It includes:
        - Importing necessary modules.
        - Data loading and preprocessing (using `readData.py` and `createData.py`).
        - Model instantiation (using `Transformer.py`).
        - Training loop (possibly calling `TrainModel.py`).
        - Evaluation and saving the trained model to `trainedModel.pth`.
        It serves as the primary entry point for experimenting with hyperparameters and visualizing training progress.

### module/bot.py
This file probably defines the chatbot interface for inference. It loads the trained model, processes user input, generates responses using the Transformer decoder and handles the conversation loop.Key functions might include `predict` or `chat` for generating replies based on input sequences.

### module/createData.py
Responsible for creating or preprocessing the dataset. It likely tokenizes text from `data.txt`, builds vocabulary, and generates input-output pairs for training, saving them to preprocessed files like `dataset.txt` or `extractedData.json`.

### module/cuda.py
A utility module for device management. It checks for CUDA availability and sets the device (GPU or CPU) for model training and inference, ensuring compatibility with hardware acceleration.

### module/decoder.py
the Decoder component of the Transformer. It stacks multiple decoder layers (`dl.py`), incorporating masked multi-head attention, cross-attention with encoder outputs, and feed-forward networks.

### module/dl.py
Defines a single Decoder Layer. It includes sublayers for self-attention (masked), cross-attention, position-wise feed-forward, and layer normalization, with residual connections.

### module/el.py
Defines a single Encoder Layer. It consists of multi-head self-attention, position-wise feed-forward network, layer normalization, and residual connections.

### module/encoder.py
the Encoder stack of the Transformer. It combines multiple encoder layers (`el.py`) with positional encoding to process input sequences.

### module/LN.py
Layer Normalization. A class or function to normalize activations across features, used in encoder and decoder layers to stabilize training.

### module/multiheadAttention.py
Defines Multi-Head Self-Attention mechanism. It splits queries, keys, and values into multiple heads, computes scaled dot-product attention in parallel, and concatenates results.

### module/multiheadCattention.py
Multi-Head Cross-Attention. Similar to self-attention but used in the decoder to attend to encoder outputs (queries from decoder, keys/values from encoder).

### module/pff.py
Positionwise FeedForward Network.A simple two layer fully connected network applied independently to each position in the sequence,used in both encoder and decoder layers.

### module/postionalEncoding.py
Positional Encoding. Adds sinusoidal encodings to input embeddings to incorporate sequence order information.

### module/readData.py
A data loading module.It reads raw data from files like `data.txt`,parses conversation pairs and prepares them for tokenization.

### module/scaledDotProduct.py
Scaled Dot-Product Attention.Computes attention scores as softmax(QK^T/sqrt(dk))*V with optional masking.

### module/se.py
Likely defines a Sublayer Connection or similar utility. It wraps sublayers with residual connections and layer normalization (x+dropout(sublayer(LN(x)))).

### module/seqDec.py
Sequential Decoding. Handles autoregressive generation during inference, feeding generated tokens back into the decoder.

### module/sequencialEncoding.py
Sequential Encoding (note: likely a typo for "sequential"). Possibly wraps the encoder with input embedding and positional encoding.

### module/sqEnc.py
Likely a shorthand for Sequence Encoder. Might be an alternative or helper for encoding input sequences.

### module/TestModel.py
Contains functions or scripts to test the trained model. It evaluates performance on held-out data, computes metrics like BLEU score, or runs sample predictions.

### module/TrainModel.py
Defines the training loop for the model.It includes optimizer setup, loss computation (cross-entropy),forward/backward passes and epoch management.

### module/Transformer.py
The core Transformer model class. It integrates the encoder, decoder, embeddings, and final projection layer for sequence-to-sequence tasks like translation or chatting.

### module/VD.py
Likely handles Vocabulary and Data utilities (e.g., Vocabulary Dictionary). Builds word-to-index mappings,handles special tokens (SOS,EOS,PAD) and possibly dataset classes for DataLoader.

### Other Files
        - **run.py**: The main script to launch the chatbot in interactive mode. It imports the bot module, loads the model and runs a loop for user input and bot responses.
        - **asset/data/data.txt**: Raw data,in tab separated format (question\tanswer).
        - **asset/preprocessed/dataset.txt**: Preprocessed text data,tokenized and cleaned versions of the raw data.
        - **asset/preprocessed/extractedData.json**: JSON-structured data, such as vocabulary, token mappings and extracted features for training.
        - **model/trainedModel.pth**: Saved PyTorch model state dictionary after training.


## Acknowledgments
Inspired by the original transformer paper "Attention is All You Need" and PyTorch tutorials on sequence models.
