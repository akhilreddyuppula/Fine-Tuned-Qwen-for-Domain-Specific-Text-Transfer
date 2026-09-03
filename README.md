# Fine-Tuned Qwen for Domain-Specific Text Transfer

A language-model project that takes the Qwen model (via Hugging Face) and applies it to style-conditioned text generation — converting plain modern English into Shakespearean / bard-style verse.

## What It Does

Given an input sentence in everyday English, the model rewrites it in the style of Shakespearean poetry, preserving meaning while transforming register, vocabulary, and phrasing.

## Approach

- Built on the **Qwen** model accessed through the **Hugging Face** ecosystem.
- Framed as a **style transfer** task: the content stays fixed while the stylistic surface changes.
- <!-- VERIFY: state whether you fine-tuned the model on a dataset, or used prompting/in-context conditioning. These are different claims — describe exactly what you did. -->

## Tech Stack

Python, Hugging Face Transformers, Qwen

## Example

<!-- Add a couple of real input → output examples from your own runs here. Real examples are the single most convincing thing in this README. -->

| Input (Modern English) | Output (Shakespearean) |
|---|---|
| _add example_ | _add example_ |



## Notes / Future Improvements

- Add quantitative evaluation of style transfer quality.
- Expand to other target styles to demonstrate generality.

<!-- SCREENSHOT: optional — a screenshot of a few real input/output pairs from the running model works well here. -->
<img width="1255" height="231" alt="25 06 2026_22 24 53_REC" src="https://github.com/user-attachments/assets/b89ce8c7-90e2-436f-8ee9-14532f6f9cf2" />
