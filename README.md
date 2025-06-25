# 💬 Sentiment Classifier with LangGraph DAG and Analytics

This project is a **CLI-based sentiment classifier** that combines:
- LoRA fine-tuning on DistilBERT using the IMDb dataset
- A LangGraph DAG (Directed Acyclic Graph) with conditional routing
- A confidence-based fallback system with user clarification
- Real-time CLI statistics and visualizations of confidence curves and fallback rates

---

## 📂 Project Structure
-langgraph_classifier
├──src
  ├── train.py # Fine-tunes the model using PEFT (LoRA)
  ├── dag.py # Defines the LangGraph DAG and analytics tracker
  ├── cli.py # CLI interface to interact with the model and the DAG
  ├── models/ # Directory where fine-tuned models are saved
  ├── logs/ # Logs and analytics history
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🚀 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Hari-1-0/Sentiment-Fallback-Classifier
   cd langgraph_classifier
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## 🏋️ Fine-Tuning Instructions
To fine-tune the model on the IMDb dataset with LoRA:

  ```bash
  python train.py
```
This will:
- Fine-tune distilbert-base-uncased using LoRA on 10,000 training samples and 2,000 testing samples.

- Save the fine-tuned model and tokenizer to the ./models/lora_finetuned directory.
