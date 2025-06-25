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
   git clone <your-repo-url>
   cd <your-project-directory>
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## 🏋️ Fine-Tuning Instructions

To fine-tune the model on the IMDb dataset with LoRA:

```bash
python train.py
