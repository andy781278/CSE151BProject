# ARC on DeBERTa-v3 with Retrieval and Reranking

CSE 151B final project, Andy Zhu. [Presentation slides](https://docs.google.com/presentation/d/1U_8r92b0rmfG-Y1q4Qvh99dELYdqJTq-PFHHedsvRaQ/edit)

## Goal
Measure how well a fine-tuned DeBERTa-v3-large answers the AI2 Reasoning Challenge (ARC), a set of about 7,800 grade-school multiple-choice science questions. Measure how much retrieval and reranking add on the Challenge split, which was built from questions that retrieval-based solvers answered wrong.

## Pipeline
1. Pair the question with each answer choice.
2. Retrieve the top 20 passages for each pair from the 14M-sentence ARC corpus with BM25.
3. Rerank the 20 passages with a cross-encoder and keep the top k.
4. Feed `[CLS] context [SEP] question [SEP] choice [SEP]` to DeBERTa-v3-large, score each choice, and take the softmax.

Other techniques: linear learning-rate warmup, training on Easy and Challenge combined (testing on Challenge only), masking padded answer slots, truncating context only, and averaging softmax outputs across seeds (ensembling).

## Results (ARC-Challenge test accuracy)

| Configuration | Mean ± std | Ensemble |
| --- | --- | --- |
| Baseline, no ensembling or combined training | 0.684 | n/a |
| No retrieval (k=0) | 0.7150 ± 0.0095 | 0.737 |
| BM25, k=3 | 0.7119 ± 0.0110 | 0.733 |
| BM25, k=10 (6 seeds) | 0.7226 ± 0.0123 | 0.744 |
| BM25 + cross-encoder reranking, k=10 (6 seeds) | **0.7399 ± 0.0050** | **0.758** |

## Findings
- DeBERTa-v3-large accounts for most of the accuracy, scoring about 71% with no retrieved context.
- BM25 retrieval alone gave no significant gain. Keyword matching mostly returns text that shares words with the question, not text that separates the correct answer from the wrong ones.
- Reranking the same 20 BM25 passages raised mean accuracy by 1.7 points (t ≈ 3.2) and cut seed variance by more than half. No new passages were retrieved, so the useful passages were already in BM25's results but ranked too low to be used.
