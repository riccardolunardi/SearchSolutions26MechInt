# Inside Language Models: From Probing to Mechanistic Interpretability

Tutorial materials for **Search Solutions Tutorial Day**, organised by the BCS Information Retrieval Specialist Group, **24 November 2026, London**.

**Presenters:** Kevin Roitero and Riccardo Lunardi, University of Udine.

See the [official event page](https://www.bcs.org/events-calendar/2026/november/search-solutions-tutorial-day/) for the programme and event details.

The tutorial introduces methods for studying the language models used in search and retrieval-augmented generation (RAG), from decoding information in hidden representations to investigating how activations affect model behaviour.

## Materials

- [Practicum I: probing hidden representations](practicum1/README.md): a self-contained notebook with saved model responses and activations. The default analysis runs without a GPU.
- [Practicum I Colab upload](practicum1/practicum1_upload.zip): input data and cached activations.
- [Practicum II: activation patching and sentiment steering](practicum2/README.md): a self-contained notebook for local/Colab GPU interventions or replay of the supplied results without a GPU.
- [Practicum II Colab upload](practicum2/practicum2_upload.zip): input data and saved intervention results.

Practicum I demonstrates reading information from activations. Practicum II changes activations: copying a state between factual-answer runs, then adding a sentiment direction during generation.

Start with the [Practicum I instructions](practicum1/README.md).

Both notebooks use the shared [requirements.txt](requirements.txt). Install it in the execution environment before running. For a local GPU model run, use CUDA-enabled PyTorch.
