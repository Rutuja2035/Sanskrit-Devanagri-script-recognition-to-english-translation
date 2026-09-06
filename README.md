# Sanskrit-Devanagri-script-recognition-to-english-translation
# 🕉️ Sanskrit Vision

### Handwritten Sanskrit → Digital Text → Translation

**Sanskrit Vision** is an end-to-end AI-powered system designed to recognize **handwritten Sanskrit in Devanagari script** and convert it into digital text that can be translated into another language.

The project combines **image processing, OCR, text cleaning, and translation** into a single pipeline with an easy-to-use **Streamlit web interface**.

---

## 🌟 Why Sanskrit Vision?

A large amount of Sanskrit knowledge exists in handwritten manuscripts and documents. Unfortunately, converting these documents into machine-readable text is difficult.

Handwritten Devanagari presents several challenges:

* Complex characters and conjuncts
* The Devanagari **Shirorekha (headline)**
* Different handwriting styles
* Noisy or low-quality document images
* Limited datasets for handwritten Sanskrit
* Limited OCR models specifically designed for Sanskrit

**Sanskrit Vision** aims to make this process simpler by bringing OCR and translation together in one application.

---

## 🚀 What Can It Do?

The system follows a simple pipeline:

```text
📷 Image
   ↓
🖼️ Image Preprocessing
   ↓
🔍 Sanskrit OCR
   ↓
🧹 Text Cleaning
   ↓
🌐 Translation
   ↓
📄 Final Output
```

### ✨ Key Features

| Feature                 | Description                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| 🖼️ Image Preprocessing | Improves document quality using grayscale, binarization, denoising, and contrast enhancement |
| 🔍 OCR                  | Extracts Sanskrit Devanagari text from images                                                |
| 🧹 Post-Processing      | Cleans and normalizes extracted Unicode text                                                 |
| 🕉️ Sanskrit Support    | Designed specifically around handwritten Sanskrit in Devanagari                              |
| 🌐 Translation          | Provides a modular translation layer                                                         |
| 🖥️ Streamlit UI        | Simple web interface for uploading images and viewing results                                |
| 🧩 Modular Architecture | OCR and translation components can be replaced or upgraded                                   |

---

# 🏗️ System Architecture

Sanskrit Vision is built as a modular pipeline so that individual components can be improved independently.

```text
                ┌──────────────────┐
                │   Sanskrit Image │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Image Processing │
                │  & Enhancement   │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │       OCR        │
                │    PaddleOCR     │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Text Processing  │
                │ & Normalization  │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │   Translation    │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │  Translated Text │
                └──────────────────┘
```

For a detailed architecture, see:

`docs/architecture/system_architecture.md`

---

# 📂 Project Structure

```text
Text-Translation/
│
└── sanskrit_ocr_translation/
    │
    ├── app.py
    │
    ├── src/
    │   ├── preprocessing/
    │   ├── ocr/
    │   ├── translation/
    │   └── postprocessing/
    │
    ├── scripts/
    │   ├── data preparation
    │   └── evaluation
    │
    ├── tests/
    │
    ├── docs/
    │   └── architecture/
    │
    ├── requirements.txt
    └── README.md
```

### Important Files

**`app.py`**
Main Streamlit application.

**`src/`**
Contains the core OCR, preprocessing, translation, and text-processing logic.

**`scripts/`**
Utility scripts for preparing data and evaluating the system.

**`tests/`**
Automated tests for different components.

**`docs/`**
Detailed technical documentation and architecture information.

---

# ⚙️ Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/AkshadKurve/Text-Translation.git
```

Move into the project directory:

```bash
cd Text-Translation/sanskrit_ocr_translation
```

---

## 2. Install Python Dependencies

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 🔍 Install PaddleOCR

Sanskrit Vision uses **PaddleOCR** with a Devanagari recognition model for Sanskrit OCR.
The first run downloads the selected model files.

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser and upload a Sanskrit manuscript or handwritten Devanagari image.

---

# 🧪 Testing

Run the test suite using:

```bash
pytest
```

This helps verify that the different components of the pipeline are working correctly.

---

# 🧠 Technology Stack

Sanskrit Vision brings together several technologies:

* **Python** — Core programming language
* **OpenCV** — Image preprocessing
* **PaddleOCR** — Sanskrit Devanagari text detection and recognition
* **Streamlit** — Web interface
* **Pytest** — Testing
* **Unicode Normalization** — Sanskrit/Devanagari text processing
* **Machine Translation** — Translation layer

The modular design makes it possible to replace individual technologies as better models become available.

---

# ⚠️ Current Limitations

Sanskrit Vision is currently a research/development project and has some limitations.

### OCR

OCR performance depends heavily on:

* Image quality
* Handwriting style
* Character complexity
* Document layout
* Availability of representative handwritten and manuscript training data

A custom deep-learning OCR model would likely improve recognition accuracy with a sufficiently large and diverse dataset.

### Translation

The current translation component provides a baseline implementation.

For production-level Sanskrit translation, a specialized neural translation model such as **IndicTrans2** or another Sanskrit-capable model could be integrated.

---

# 🛣️ Future Improvements

Some planned improvements include:

* [ ] Train a custom handwritten Sanskrit OCR model
* [ ] Add a larger Sanskrit handwriting dataset
* [ ] Improve compound-character recognition
* [ ] Improve Shirorekha detection
* [ ] Add better document segmentation
* [ ] Integrate a production-grade translation model
* [ ] Add confidence scores for OCR predictions
* [ ] Support PDF and multi-page documents
* [ ] Add side-by-side original and translated text
* [ ] Improve Sanskrit-specific text correction
* [ ] Add model evaluation metrics
* [ ] Deploy the application online

---

# 🎯 Project Goal

The long-term goal of **Sanskrit Vision** is to help make handwritten Sanskrit documents more accessible by creating a pipeline that can:

> **See → Read → Understand → Translate**

By combining computer vision, OCR, natural language processing, and machine translation, the project aims to contribute to the **digitization and accessibility of Sanskrit literature**.

---

# 🤝 Contributing

Contributions and ideas are welcome!

If you would like to improve the project:

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/my-improvement
```

3. Make your changes
4. Commit them

```bash
git commit -m "Add my improvement"
```

5. Push your branch

```bash
git push origin feature/my-improvement
```

6. Open a Pull Request

---

# 📜 License

Add your project's license information here.

If you haven't selected a license yet, consider choosing an appropriate open-source license before publishing the project publicly.

---

# 👨‍💻 Author

**Akshad Kurve and Rutuja Deshmukh** 

---

## ⭐ Support the Project

If you find **Sanskrit Vision** interesting or useful, consider giving the repository a ⭐ on GitHub.

Every star helps the project reach more developers and researchers interested in **Sanskrit, OCR, computer vision, and natural language processing**.

---

### 🕉️ Sanskrit Vision

**Preserving ancient knowledge with modern AI.**
