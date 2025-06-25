from langgraph.graph import END, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import logging

model_path = "./models/lora_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

CONFIDENCE_THRESHOLD = 0.7

logger = logging.getLogger("LangGraphLogger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./logs/pipeline.log")
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def inference_node(state):
    text = state["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label = "Positive" if predicted_class.item() == 1 else "Negative"
    confidence_value = confidence.item()

    logger.info(f"[InferenceNode] Input: {text} | Predicted: {label} | Confidence: {confidence_value:.2f}")

    state["prediction"] = label
    state["confidence"] = confidence_value
    return state

def confidence_check_node(state):
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        logger.info(f"[ConfidenceCheckNode] Confidence: {state['confidence']:.2f} (Accepted)")
    else:
        logger.info(f"[ConfidenceCheckNode] Confidence: {state['confidence']:.2f} (Fallback Triggered)")
    return state

def fallback_node(state):
    print(f"\n[FallbackNode] Could you clarify your intent? Was this a negative review?")
    user_feedback = input("User: ").strip().lower()

    if "yes" in user_feedback:
        corrected_label = "Negative"
    elif "no" in user_feedback:
        corrected_label = "Positive"
    else:
        corrected_label = "Unclear"

    logger.info(f"[FallbackNode] User clarified: {user_feedback} | Corrected Label: {corrected_label}")

    state["final_label"] = corrected_label
    return state

def accept_node(state):
    logger.info(f"[AcceptNode] Final label accepted: {state['prediction']}")
    state["final_label"] = state["prediction"]
    return state

def route_after_confidence_check(state):
    """Router function to determine next node based on confidence check result"""
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return "AcceptNode"
    else:
        return "FallbackNode"

def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("InferenceNode", inference_node)
    workflow.add_node("ConfidenceCheckNode", confidence_check_node)
    workflow.add_node("FallbackNode", fallback_node)
    workflow.add_node("AcceptNode", accept_node)

    workflow.set_entry_point("InferenceNode")
    workflow.add_edge("InferenceNode", "ConfidenceCheckNode")

    workflow.add_conditional_edges(
        "ConfidenceCheckNode",
        route_after_confidence_check,
        {
            "AcceptNode": "AcceptNode",
            "FallbackNode": "FallbackNode"
        }
    )

    workflow.add_edge("FallbackNode", END)
    workflow.add_edge("AcceptNode", END)

    return workflow.compile()