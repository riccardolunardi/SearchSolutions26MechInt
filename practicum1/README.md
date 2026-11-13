# Practicum I: probing correct and counterfactual passages

Part of **Inside Language Models: From Probing to Mechanistic Interpretability**, by Kevin Roitero and Riccardo Lunardi (University of Udine), at [Search Solutions Tutorial Day](https://www.bcs.org/events-calendar/2026/november/search-solutions-tutorial-day/), 24 November 2026.

## What this demonstrates

Can a linear classifier distinguish a correct passage from a counterfactual passage using a language model's activations **before it answers**?

We train a separate probe at every layer and compare their performance on held-out questions. The aim is just to illustrate probing and its interpretation.

## Run the notebook

All experiment code is in [practicum1.ipynb](practicum1.ipynb). The default setting, `USE_SAVED_DATA = True`, uses the supplied responses and activations. **No GPU or model download is needed for this mode.** Internet access is needed to install dependencies if they are not already available.

### Google Colab

1. Download `practicum1.zip` from the tutorial materials and unzip it on your computer.
2. Open [Google Colab](https://colab.research.google.com/) and upload `practicum1.ipynb` from the extracted files.
3. Use a CPU runtime and leave `USE_SAVED_DATA = True`.
4. Run the cells in order. When the data-loading cell requests a file, upload the original `practicum1.zip`.

### Local Jupyter

Unzip the package, open `practicum1.ipynb` in Jupyter, and run the cells in order. Keep the notebook and its `data/` folder together, and use that directory as the kernel's working directory. The dependency cell installs NumPy, scikit-learn, and Matplotlib into the notebook environment.

## Data and procedure

The model is **Qwen/Qwen2.5-1.5B-Instruct**. The supplied activations were collected in bfloat16 on a Google Colab GPU.

1. Ask 50 factual questions without a passage. Retain the **47 questions answered correctly**.
2. For each retained question, present two separate prompts: one with a correct passage and one with a counterfactual passage. This gives **94 examples**: label `0` for correct, `1` for counterfactual.
3. Extract the last prompt token's activations at each of the 29 returned hidden states, before generating any answer tokens. Each vector has 1,536 values. State `0` is the embedding output.
4. Split by question into **37 training questions and 10 test questions**. Both passages for a question stay together. The split uses category stratification and random seed `42`.
5. At each layer, fit a standardized linear logistic regression probe with L2 regularization, using both passage classes in the training set. Evaluate on the test set and plot balanced accuracy across layers.

Baselines use the same split: majority class, length/overlap features, word TF-IDF, and character TF-IDF. Vocabulary and TF-IDF weights are also learned only from training examples.

Both passage conditions use this instruction:

```text
Answer the question as accurately as possible. The accompanying passage may contain factual errors. Give only the answer entity or value, without explanation.

Passage: {passage}

Question: {question}
```

The closed-book condition asks for an accurate answer without mentioning a passage.

## How to interpret the results

In the supplied run, the model answers 47/50 questions correctly without context, but follows the passage's answer in all 50 correct-passage trials and all 50 counterfactual-passage trials, despite the warning.

A successful probe shows that information useful for distinguishing the passage labels is **linearly decodable** from the activations. It does not prove that the model recognises a falsehood, has an explicit internal truth variable, or uses the decoded information when answering. Closed-book success is an observed correct answer, not proof of stable internal knowledge.

## Regenerate the model run (optional)

On Google Colab, use a GPU runtime and set `USE_SAVED_DATA = False`. Run all cells to regenerate prompts, token IDs, answers, and activations, then fit the probes. This mode downloads the model and installs Transformers and Accelerate. Different hardware or numerical precision can change responses and results.

The final cell exports `train_test_results.zip`, containing the regenerated data, split assignments, per-layer predictions, scores, and baselines. In Colab, it starts a download. The supplied input data files remain unchanged.

## Files

| File | Purpose |
|---|---|
| `practicum1.ipynb` | Complete experiment and explanations |
| `data/pilot.json` | Model metadata, questions, prompts, token IDs, and responses |
| `data/activations.npz` | Saved prompt-only hidden states and example/model identifiers |
| `README.md` | Setup and interpretation guide |