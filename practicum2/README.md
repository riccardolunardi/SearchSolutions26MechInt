# Practicum II: activation patching and sentiment steering

Part of **Inside Language Models: From Probing to Mechanistic Interpretability**, by Kevin Roitero and Riccardo Lunardi (University of Udine), at [Search Solutions Tutorial Day](https://www.bcs.org/events-calendar/2026/november/search-solutions-tutorial-day/), 24 November 2026.

## Setup

Locally, keep the notebook and its `data/` directory together, and use that directory as the kernel's working directory. In Colab, upload `practicum2_upload.zip` when requested. Install dependencies in the execution environment before running:

```bash
python -m pip install -r ../requirements.txt
```

For replay, open `practicum2.ipynb` with `RUN_MODEL = False` and use a CPU runtime. In that case, no model download is needed. The upload ZIP contains data and saved results, not the notebook.

For a local GPU model run, use CUDA-enabled PyTorch. Set `RUN_MODEL = True` and `DEVICE = "auto"`, or specify a GPU such as `DEVICE = "cuda:1"`, then run all cells. CPU inference is also possible but slow. In Colab, use a GPU runtime only for `RUN_MODEL = True`, and upload `practicum2_upload.zip` when asked.

The model is `Qwen/Qwen2.5-1.5B-Instruct`, as in Practicum I. The first run downloads the checkpoint. The final cell exports `practicum2_upload.zip`.

## Activation patching: compare two locations

Run correct-passage donors and counterfactual-passage recipients for 12 questions. Copy a donor activation into the recipient at either:

- **The changed entity in the passage:** for example, transfer the Canberra state into the Melbourne position. If an entity spans several tokens, consistently patch its last subtoken.
- **The final prompt token:** the last position, including the assistant prefix added by the chat template.

One plot with two evaluation curves shows the change in preference for the correct versus false answer, using complete answer strings' summed conditional log probabilities. The plot shows evaluation effects on eight questions.

## Sentiment steering

Construct a mean positive-minus-negative vector from 24 matched review-sentence pairs, reading the final token of `Review: {sentence}`. Choose among three block candidates using restaurant/product development prompts, then keep that block fixed for hotel, film, museum, and library prompts.

Generate reviews at strengths -2, -1, 0, +1, and +2. Strength 1 adds the raw mean-difference vector at the final prompt position and subsequent answer positions. The prompt, model weights, and greedy decoding stay fixed.

Read the generated reviews alongside the candidate-preference curve: when the mean-difference vector is added, the model's preference for the positive candidate increases, and the generated review becomes more positive. If the vector is subtracted, the preference decreases and the review becomes more negative. The plot shows evaluation effects on 12 questions.

## Replay saved results

Set `RUN_MODEL = False` and `IMPORT_RESULTS = True` to use the existing unpacked results. In Colab, you'll have to upload `practicum2_upload.zip` to reuse the data. Otherwise, if you want to run it locally, the notebook reads the folder specified in `RESULTS_ARCHIVE`. Alternatively, to use the notebook locally, extract `results.json` and `steering_vectors.npz` into a `results/` directory beside the notebook.