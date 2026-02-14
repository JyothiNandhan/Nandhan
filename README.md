
# 🧠 NANO-VLM: Mini Vision-Language Model from Scratch

This notebook implements a **lightweight Vision-Language Model (VLM)** trained on a fully synthetic dataset of colored shapes.
It demonstrates how CLIP-style contrastive learning works using a minimal, educational pipeline.

The goal is to understand how image–text alignment models function internally — without relying on large pretrained transformers.

---

# 📌 Project Overview

This notebook:

* Generates a **synthetic dataset** of images containing colored shapes
* Automatically creates corresponding **text captions**
* Builds:

  * An **Image Encoder (CNN)**
  * A **Text Encoder (Embedding + Projection)**
* Trains them using a **CLIP-style contrastive loss**
* Enables:

  * Embedding visualization
  * Text-to-image retrieval

This is a small-scale educational replica of models like CLIP, but implemented from scratch in PyTorch.

---

# 🏗️ Notebook Structure

## 1️⃣ Setup & Imports

* Imports PyTorch, NumPy, math, random
* Sets device (CPU/GPU)

---

## 2️⃣ Synthetic Dataset Generation

### Dataset Properties

* Colors: red, green, blue, yellow, purple, orange, pink, brown, gray
* Shapes: (e.g., circle, square, triangle)
* Random positions

### Image Creation

The function:

```python
draw_sample(color, shape, position)
```

Generates synthetic images of colored shapes.

---

## 3️⃣ Custom Dataset Class

```python
class ShapesDataset()
```

Responsible for:

* Generating image–caption pairs
* Tokenizing captions
* Building vocabulary
* Encoding text numerically

Dataset is split into:

* 80% Training
* 20% Validation

DataLoader is used for batching.

---

## 4️⃣ Model Architecture

## 🖼️ Image Encoder

```python
class ImageEncoder(nn.Module)
```

* Convolutional Neural Network
* Extracts image features
* Projects features into embedding space
* Output: fixed-dimensional image embedding

---

## 📝 Text Encoder

```python
class TextEncoder(nn.Module)
```

* Token embedding layer
* Mean pooling / simple aggregation
* Projection to same embedding space as image encoder
* Output: fixed-dimensional text embedding

---

## 🔥 Contrastive Loss (CLIP-Style)

```python
clip_loss(img_emb, txt_emb)
```

Implements:

* Cosine similarity
* Temperature scaling
* Cross-entropy over similarity matrix

This aligns:

* Correct image–text pairs → high similarity
* Incorrect pairs → low similarity

---

## 5️⃣ Training Pipeline

* Initialize encoders
* Define optimizer
* Train over batches
* Minimize contrastive loss
* Periodically evaluate

---

## 6️⃣ Visualization Utilities

### Show Sample Images

```python
show_image()
```

### Embedding Visualization

```python
visualize_results_with_embeddings()
```

Helps inspect:

* Alignment between text and image embeddings
* Similarity structure

---

## 7️⃣ Text-to-Image Retrieval

```python
text_to_image_retrieval(query_text)
```

Given a text query:

* Encode text
* Compute similarity with all image embeddings
* Retrieve most similar images

This demonstrates cross-modal retrieval.

---

# 🧪 What This Project Demonstrates

✔ How vision-language models align modalities
✔ How contrastive learning works
✔ How embedding spaces are formed
✔ How retrieval systems operate
✔ How CLIP-like training functions internally

---

# 🛠 Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

Run inside:

* Google Colab (recommended)
* Local environment with GPU (optional but helpful)

---

# ▶️ How to Run

1. Open in Google Colab
2. Run cells sequentially
3. Train model
4. Try:

```python
text_to_image_retrieval("red square")
```

---

# 🎯 Learning Goals

This notebook is ideal for:

* Understanding CLIP intuitively
* Learning multimodal embedding alignment
* Building VLMs from scratch
* Research prototyping before scaling

---

# 🚀 Possible Extensions

* Replace simple text encoder with Transformer
* Add positional embeddings
* Increase dataset complexity
* Add attention mechanism
* Scale to real-world datasets
* Replace CNN with Vision Transformer

---

# 📚 Conceptual Summary

This notebook implements:

[
\text{Image Encoder} \rightarrow \mathbf{z}_i
]
[
\text{Text Encoder} \rightarrow \mathbf{z}_t
]

Then optimizes:

[
\text{Similarity}(\mathbf{z}_i, \mathbf{z}_t)
]

So that matching pairs are close in embedding space.

---

# 🧩 Why "NANO-VLM"?

Because this is a **minimal educational version** of large vision-language models like CLIP, built entirely from scratch and small enough to fully understand.

