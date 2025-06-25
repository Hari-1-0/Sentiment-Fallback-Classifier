from dag import build_graph

def main():
    print("Welcome to the Sentiment Classifier with Fallback!")
    print("Type 'exit' to quit.\n")

    graph = build_graph()

    while True:
        user_input = input("Enter your review: ").strip()
        if user_input.lower() == "exit":
            break

        initial_state = {"input": user_input}

        final_state = graph.invoke(initial_state)

        final_label = final_state.get('final_label', 'No label found')
        confidence = final_state.get('confidence', 0.0)
        print(f"\nFinal Decision: {final_label} (Confidence: {confidence:.2f})\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
