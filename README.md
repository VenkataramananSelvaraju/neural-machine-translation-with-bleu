# NMT System with Automatic Evaluation
A specialized Neural Machine Translation (NMT) application built to translate medical text and evaluate accuracy using the industry-standard BLEU metric.

# Group No: 73
**Group Member Names:**
* DEEPESH PARMAR (2024AA05053) 100%
* PRAJAPATI HEMANG KEYURBHAI (2024AA05058) 100%
* SATISH KUMAR PATHAK (2024AA05578) 100%
* V. NIKITHA (2024AA05552) 100%
* VENKATARAMANAN S (2024AA05555) 100%

## Repository URL: 
https://github.com/VenkataramananSelvaraju/neural-machine-translation-with-bleu

## 🚀 Features
* **Transformer-Based Translation:** Uses the `MarianMT` architecture (Helsinki-NLP) for high-context medical translation.
* **Automated BLEU Scoring:** Computes translation quality scores instantly.
* **N-Gram Analysis:** Provides a detailed precision breakdown (1-gram to 4-gram) to analyze vocabulary vs. fluency.
* **Multi-Reference Support:** Allows multiple human reference translations to account for medical synonyms.
* **Brevity Penalty Detection:** Automatically flags if translations are too short compared to the reference.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **NMT Engine:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* **Metrics:** [SacreBLEU](https://github.com/mjpost/sacrebleu)
* **Language:** Python 3.14+

## 📦 Installation

1. **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <project-folder-name>
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

1. **Run the application:**
    ```bash
    streamlit run app.py
    ```
2. **Input Text:** Enter a medical sentence in English (e.g., "The patient was diagnosed with acute pneumonia.").
3. **Add References:** Enter the expected French translation. You can enter multiple versions on new lines.
4. **Analyze:** Click **"Translate & Evaluate"** to view the NMT output, BLEU score, and precision table.

## 📊 Understanding the Scores
* **1-Gram:** Measures individual word accuracy (Medical vocabulary).
* **4-Gram:** Measures phrase fluency and sentence structure.
* **Brevity Penalty (BP):** If $BP < 1.0$, the model's output was too short, which may indicate missing clinical details.

## 📂 Project Structure
* `app.py`: Main application code (Streamlit + Transformers).
* `Report.pdf`: Technical documentation on design choices and strategies.
* `medical_data.csv`: Sample dataset for testing.
