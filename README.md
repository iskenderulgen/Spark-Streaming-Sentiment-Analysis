# Spark-Streaming-Sentiment-Analysis

Spark Structured Streaming project for real‑time text classification.  
Originally built on the Twitter API; the first release consumed live tweets. Since Twitter's API access has changed, this version reads from local Parquet files (`data/yelp/`) for streaming.

## Features

- Stream ingestion via Spark Structured Streaming (Parquet source)
- Two classification pipelines:
  - **Spark‑NLP**: uses DocumentAssembler, Tokenizer, GloVe embeddings & LogisticRegression
  - **scikit‑learn**: standalone GloVe reader, NumPy embeddings & LogisticRegression
- Sentiment visualization: batch‑wise line and pie charts with Matplotlib
- Balanced sampling of IMDB dataset for offline training (`data/train‑imdb.parquet`)
- Modular notebooks for experimentation and deployment
- Real-time sentiment trend analysis with dynamic visualization
- Scalable architecture suitable for distributed deployment

## Technical Architecture

The project implements two distinct NLP pipelines with the same functional goal:

1. **Spark-NLP Pipeline**:
   - Uses John Snow Labs' Spark-NLP library for distributed NLP processing
   - DocumentAssembler → Tokenizer → WordEmbeddings → SentenceEmbeddings → EmbeddingsFinisher
   - Leverages Spark ML LogisticRegression for distributed training and prediction
   - Optimized for large-scale processing with built-in parallelism

2. **scikit-learn Pipeline**:
   - Lightweight implementation using NumPy and scikit-learn
   - Custom tokenization → GloVe embeddings → Sentence averaging → LogisticRegression
   - Pandas integration for simplified data manipulation within Spark context
   - Improved development experience with familiar Python data science stack

Both pipelines share the same streaming infrastructure and visualization components.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── .gitignore
├── data/
│   ├── train‑imdb.parquet       ← IMDB sentiment dataset
│   ├── glove_100d/              ← GloVe 100d for Spark‑NLP
│   ├── glove.6B.50d.txt         ← GloVe 50d for scikit‑learn
│   └── yelp/                    ← streaming source (Parquet files)
├── spark_streaming_sparknlp.ipynb
└── spark_streaming_sklearn.ipynb
```

## Prerequisites

- Java 8+  
- Python 3.7+  
- Apache Spark 3.3.1
- Spark‑NLP 5.5.3

## Data Preparation

- **Offline IMDB**: `data/train‑imdb.parquet`   [Stanford IMDB Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) - Used for training the model
- **GloVe embeddings**:  
  - Spark‑NLP: unpack/download into `data/glove_100d/`  
  - scikit‑learn: place `glove.6B.50d.txt` in `data/`  
- **Streaming source**: drop Parquet files into `data/yelp/` (one file per trigger) [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full) - Used for testing (file is divided into 10k row parts for streaming)
### Spark‑NLP Pipeline

1. Open `spark_streaming_sparknlp.ipynb`  
2. Adjust paths/configs if needed  
3. Run all cells to train on IMDB and start structured streaming on `data/yelp/`

### scikit‑learn Pipeline

1. Open `spark_streaming_sklearn.ipynb`  
2. Run offline training cells  
3. Start streaming cells (`process_batch`) on `data/yelp/`

## License

This project is licensed under Apache License 2.0 – see the [LICENSE](LICENSE) file.

## Acknowledgments

- John Snow Labs Spark‑NLP  
- Stanford GloVe embeddings  
- Hugging Face IMDB dataset  
- Apache Spark community

# References
This work is a simplified version of the following research

```
@inproceedings{8404540,
  author={Hakdağli, Özlem and Özcan, Caner and Oğul, Ískender Ülgen},
  booktitle={2018 26th Signal Processing and Communications Applications Conference (SIU)}, 
  title={Stream text data analysis on twitter using apache spark streaming}, 
  year={2018},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SIU.2018.8404540}}
```