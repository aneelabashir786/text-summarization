# Intelligent Text Summarization with T5

An AI system that compresses long news articles into concise, coherent summaries using an Encoder–Decoder architecture. This transforms the way information is consumed in real time.

## Project Overview

Automatic text summarization is essential for quickly digesting large volumes of information. This project leverages T5 (Text-to-Text Transfer Transformer) to generate abstractive summaries of news articles, producing human-like summaries that capture meaning beyond simple extraction.

Problem Statement

Generate concise summaries of long documents or articles using an Encoder–Decoder model.

Dataset:
Newspaper Text Summarization Dataset (CNN/DailyMail) on Kaggle

Objective:
Fine-tune an Encoder–Decoder model to perform abstractive text summarization of input passages.

## Key Features

Fine-tuned T5-Base for abstractive summarization

Preprocessing of 287K+ article-summary pairs ensuring clean, consistent data

Training optimized with AdamW, warm-up scheduling, weight decay, and fixed random seeds

Mixed-precision (FP16) GPU training for faster convergence

Evaluation metrics: ROUGE-1, ROUGE-2, ROUGE-L, along with qualitative comparison

Example outputs showing original articles and generated summaries

## Performance

Metric	Score
ROUGE-1	24.53
ROUGE-2	11.95
ROUGE-L	20.43
Validation Loss	↓ from 1.873 → 1.870

-> Highlights:

Context-aware, fluent summaries

Abstractive understanding beyond extractive methods

Reliable performance on real-world datasets

Efficient fine-tuning for real-time deployment

-> Tech Stack

Language & Libraries: Python, PyTorch

NLP Framework: Hugging Face Transformers (T5)

Web App / Demo: Streamlit (optional for live demo)

## Future Work

Domain-specific summarization for research, news, and corporate documents

Integration with Explainable AI for understanding summary generation

## Try It Out

Live Demo : https://text-summarization-zkkfoa9jbg8br7egla6roy.streamlit.app/

Read Full Details : https://medium.com/@naeemubeen639/fine-tuning-transformer-architectures-for-real-time-nlp-applications-7d320f8ee9a0
