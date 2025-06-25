from langgraph.graph import END, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import logging
import json
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

class AnalyticsTracker:
    def __init__(self):
        self.session_data = {
            "confidence_scores": [],
            "predictions": [],
            "final_labels": [],
            "fallback_triggers": [],
            "timestamps": [],
            "inputs": []
        }
        self.stats_file = "./logs/analytics_stats.json"
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical analytics data if it exists"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.historical_data = json.load(f)
            except:
                self.historical_data = {"sessions": []}
        else:
            self.historical_data = {"sessions": []}
    
    def track_prediction(self, input_text, prediction, confidence, final_label, fallback_used):
        """Track a single prediction"""
        self.session_data["confidence_scores"].append(confidence)
        self.session_data["predictions"].append(prediction)
        self.session_data["final_labels"].append(final_label)
        self.session_data["fallback_triggers"].append(fallback_used)
        self.session_data["timestamps"].append(datetime.now().isoformat())
        self.session_data["inputs"].append(input_text[:50] + "..." if len(input_text) > 50 else input_text)
    
    def save_session_data(self):
        """Save current session data to historical data"""
        if not self.session_data["confidence_scores"]:
            return
            
        session_summary = {
            "session_start": self.session_data["timestamps"][0] if self.session_data["timestamps"] else None,
            "session_end": datetime.now().isoformat(),
            "total_predictions": len(self.session_data["confidence_scores"]),
            "fallback_count": sum(self.session_data["fallback_triggers"]),
            "avg_confidence": np.mean(self.session_data["confidence_scores"]) if self.session_data["confidence_scores"] else 0,
            "data": self.session_data.copy()
        }
        
        self.historical_data["sessions"].append(session_summary)
        
        if len(self.historical_data["sessions"]) > 100:
            self.historical_data["sessions"] = self.historical_data["sessions"][-100:]
        
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2)
    
    def plot_confidence_curve(self):
        """Plot confidence scores over the current session"""
        if not self.session_data["confidence_scores"]:
            print("No data to plot yet.")
            return
        
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        confidence_scores = self.session_data["confidence_scores"]
        x_values = range(1, len(confidence_scores) + 1)
        
        plt.plot(x_values, confidence_scores, 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        
        fallback_points = [(i+1, conf) for i, (conf, fallback) in enumerate(zip(confidence_scores, self.session_data["fallback_triggers"])) if fallback]
        if fallback_points:
            fallback_x, fallback_y = zip(*fallback_points)
            plt.scatter(fallback_x, fallback_y, color='red', s=100, marker='x', label='Fallback Triggered', zorder=5)
        
        plt.xlabel('Prediction Number')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Scores Over Session')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.subplot(1, 2, 2)
        plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./logs/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confidence analysis saved to ./logs/confidence_analysis.png")
    
    def display_fallback_stats(self):
        """Display fallback frequency statistics"""
        if not self.session_data["confidence_scores"]:
            print("No data available for analysis.")
            return
        
        total_predictions = len(self.session_data["confidence_scores"])
        fallback_count = sum(self.session_data["fallback_triggers"])
        fallback_rate = (fallback_count / total_predictions) * 100 if total_predictions > 0 else 0
        avg_confidence = np.mean(self.session_data["confidence_scores"])
        
        print("\n" + "="*60)
        print("📊 CURRENT SESSION STATISTICS")
        print("="*60)
        print(f"Total Predictions: {total_predictions}")
        print(f"Fallback Triggers: {fallback_count}")
        print(f"Fallback Rate: {fallback_rate:.1f}%")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        
        self._display_confidence_histogram()
        
        if len(self.historical_data["sessions"]) > 1:
            self._display_historical_stats()
    
    def _display_confidence_histogram(self):
        """Display ASCII histogram of confidence scores"""
        if not self.session_data["confidence_scores"]:
            return
        
        print("\n📈 CONFIDENCE DISTRIBUTION (Current Session)")
        print("-" * 50)
        
        scores = self.session_data["confidence_scores"]
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(scores, bins=bins)
        
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = 40
        
        for i in range(len(bins)-1):
            bin_start = bins[i]
            bin_end = bins[i+1]
            count = hist[i]
            
            bar_length = int((count / max_count) * bar_width)
            bar = "█" * bar_length
            
            threshold_marker = " ← THRESHOLD" if bin_start <= CONFIDENCE_THRESHOLD < bin_end else ""
            
            print(f"{bin_start:.1f}-{bin_end:.1f}: {bar:<{bar_width}} ({count:2d}){threshold_marker}")
    
    def _display_historical_stats(self):
        """Display historical session statistics"""
        print("\n📈 HISTORICAL STATISTICS")
        print("-" * 50)
        
        sessions = self.historical_data["sessions"]
        total_sessions = len(sessions)
        
        total_predictions = sum(s["total_predictions"] for s in sessions)
        total_fallbacks = sum(s["fallback_count"] for s in sessions)
        avg_fallback_rate = (total_fallbacks / total_predictions * 100) if total_predictions > 0 else 0

        session_fallback_rates = []
        for session in sessions:
            if session["total_predictions"] > 0:
                rate = (session["fallback_count"] / session["total_predictions"]) * 100
                session_fallback_rates.append(rate)
        
        print(f"Total Sessions: {total_sessions}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Overall Fallback Rate: {avg_fallback_rate:.1f}%")
        
        if session_fallback_rates:
            print(f"Min Session Fallback Rate: {min(session_fallback_rates):.1f}%")
            print(f"Max Session Fallback Rate: {max(session_fallback_rates):.1f}%")
            print(f"Avg Session Fallback Rate: {np.mean(session_fallback_rates):.1f}%")
        
        print("\n🔄 RECENT SESSIONS FALLBACK RATES:")
        recent_sessions = sessions[-10:]
        for i, session in enumerate(recent_sessions, 1):
            rate = (session["fallback_count"] / session["total_predictions"] * 100) if session["total_predictions"] > 0 else 0
            bar_length = int(rate / 5)
            bar = "█" * bar_length
            print(f"Session {len(sessions)-len(recent_sessions)+i:2d}: {bar:<20} {rate:5.1f}% ({session['fallback_count']}/{session['total_predictions']})")

analytics_tracker = AnalyticsTracker()

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
    state["fallback_used"] = True
    
    analytics_tracker.track_prediction(
        state["input"], 
        state["prediction"], 
        state["confidence"], 
        corrected_label, 
        True
    )
    
    return state

def accept_node(state):
    logger.info(f"[AcceptNode] Final label accepted: {state['prediction']}")
    state["final_label"] = state["prediction"]
    state["fallback_used"] = False
    
    analytics_tracker.track_prediction(
        state["input"], 
        state["prediction"], 
        state["confidence"], 
        state["prediction"], 
        False
    )
    
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