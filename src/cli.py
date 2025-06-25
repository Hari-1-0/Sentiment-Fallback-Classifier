from dag import build_graph, analytics_tracker
import atexit

def display_help():
    """Display available commands"""
    print("\nAVAILABLE COMMANDS:")
    print("  • Enter review text - Get sentiment prediction")
    print("  • 'stats' - Show current session statistics")
    print("  • 'plot' - Generate confidence curve plot")
    print("  • 'help' - Show this help message")
    print("  • 'exit' - Quit application")
    print("-" * 50)

def main():
    print("Welcome to the Enhanced Sentiment Classifier with Analytics!")
    print("Type 'help' for available commands or 'exit' to quit.\n")
    
    atexit.register(analytics_tracker.save_session_data)
    
    graph = build_graph()
    
    display_help()

    while True:
        user_input = input("\nEnter your review (or command): ").strip()
        
        if user_input.lower() == "exit":
            print("\nSaving session data and exiting...")
            analytics_tracker.save_session_data()
            break
        elif user_input.lower() == "help":
            display_help()
            continue
        elif user_input.lower() == "stats":
            analytics_tracker.display_fallback_stats()
            continue
        elif user_input.lower() == "plot":
            try:
                analytics_tracker.plot_confidence_curve()
            except ImportError:
                print("Matplotlib not available. Install with: pip install matplotlib")
            except Exception as e:
                print(f"Error generating plot: {e}")
            continue
        elif not user_input:
            print("Please enter some text or a command.")
            continue

        initial_state = {"input": user_input}
        
        try:
            final_state = graph.invoke(initial_state)
            
            final_label = final_state.get('final_label', 'No label found')
            confidence = final_state.get('confidence', 0.0)
            fallback_used = final_state.get('fallback_used', False)
            
            status_emoji = "🔄" if fallback_used else "✅"
            confidence_bar = "█" * int(confidence * 20)
            
            print(f"\n{status_emoji} RESULT:")
            print(f"   Final Decision: {final_label}")
            print(f"   Confidence: {confidence:.3f} {confidence_bar}")
            print(f"   Fallback Used: {'Yes' if fallback_used else 'No'}")
            
            total_predictions = len(analytics_tracker.session_data["confidence_scores"])
            if total_predictions > 0 and total_predictions % 5 == 0:
                fallback_count = sum(analytics_tracker.session_data["fallback_triggers"])
                fallback_rate = (fallback_count / total_predictions) * 100
                print(f"\n📊 Quick Stats: {total_predictions} predictions, {fallback_rate:.1f}% fallback rate")
            
        except Exception as e:
            print(f"Error processing review: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()