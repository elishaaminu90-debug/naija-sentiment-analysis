from transformers import pipeline

# Load the sentiment analysis pipeline
analyzer = pipeline("sentiment-analysis")

while True:
    text = input("\nEnter a sentence to analyze (or type 'exit' to quit): ")

    if text.lower() == "exit":
        print("Goodbye!")
        break

    result = analyzer(text)[0]
    label = result['label']
    score = result['score']

    print(f"\nSentiment: {label}")
    print(f"Confidence: {score:.2f}")