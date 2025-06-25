# 💬 Sentiment Classifier with LangGraph DAG and Analytics

This project is a **CLI-based sentiment classifier** that combines:
- LoRA fine-tuning on DistilBERT using the IMDb dataset
- A LangGraph DAG (Directed Acyclic Graph) with conditional routing
- A confidence-based fallback system with user clarification
- Real-time CLI statistics and visualizations of confidence curves and fallback rates

---

## 📂 Project Structure
-langgraph_classifier/<br />
├──src<br />
  ├── train.py # Fine-tunes the model using PEFT (LoRA)<br />
  ├── dag.py # Defines the LangGraph DAG and analytics tracker<br />
  ├── cli.py # CLI interface to interact with the model and the DAG<br />
  ├── models/ # Directory where fine-tuned models are saved<br />
  ├── logs/ # Logs and analytics history<br />
├── requirements.txt # Python dependencies<br />
└── README.md # Project documentation<br />

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
## 🧩 Running the LangGraph DAG
The sentiment classification workflow is managed by a LangGraph DAG with the following nodes:

- InferenceNode: Predicts sentiment and confidence.

- ConfidenceCheckNode: Decides to accept the prediction or trigger fallback based on confidence.

- FallbackNode: Asks the user to manually clarify if confidence is low.

- AcceptNode: Accepts the model’s prediction directly if confidence is sufficient.

To launch the CLI and interact with the LangGraph DAG:
```bash
python cli.py
```
## 🖥️ CLI Commands and Workflow
When the CLI starts, you can:

- Enter a review directly → The system will predict sentiment and confidence.

- stats → Show current session fallback statistics and a CLI-based confidence histogram.

- plot → Generate a matplotlib plot of confidence curves and fallback distribution.

- help → Display the available commands.

- exit → Save analytics and exit.
