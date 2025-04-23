from Retrieval import get_top_k_courses
import numpy as np

# Defining the test set
test_queries = [
    {
        "query": "learn java programming",
        "relevant_titles": [
            "LEARN CORE JAVA PROGRAMMING ONLINE",
            "Java Project-Based Course",
            "Minecraft Java Programming for Beginners"
        ]
    },
    {
        "query": "python projects for kids",
        "relevant_titles": [
            "Build a Password generator using Python Bootcamp for kids",
            "Build a Chatbot using Python Bootcamp for kids",
            "Python Playground: Create a Rock, Paper, Scissors Game"
        ]
    },
    {
        "query": "introduction to artificial intelligence",
        "relevant_titles": [
            "Introduction to Artificial Intelligence",
            "From Beginner to AI Pro: Kickstart Your Journey With Artificial Intelligence! (For Kids)",
            "Machine Learning: 7-Day Project-Based Summer Camp"
        ]
    },
    {
        "query": "build a game using python",
        "relevant_titles": [
            "Game development using python",
            "Python Playground: Create a Rock, Paper, Scissors Game"
        ]
    },
    {
        "query": "web development course for beginners",
        "relevant_titles": [
            "Web Development from scratch",
            "HTML, CSS, JavaScript: 7-Day Summer Bootcamp"
        ]
    },
    {
        "query": "ai course for finance",
        "relevant_titles": [
            "AI in Stock Market Success: Career Growth Camp"
        ]
    },
    {
        "query": "intermediate web development",
        "relevant_titles": [
            "Web Development Pro: Intermediate Level"
        ]
    },
    {
        "query": "creative writing with ai",
        "relevant_titles": [
            "AI Pro: Creative Writing Camp for Adults"
        ]
    },
    {
        "query": "ai career growth course",
        "relevant_titles": [
            "AI Pro Camp: Career Growth Catalyst"
        ]
    },
    {
        "query": "html css javascript bootcamp",
        "relevant_titles": [
            "HTML, CSS, JavaScript: 7-Day Summer Bootcamp"
        ]
    },
    {
        "query": "coding summer camp for children",
        "relevant_titles": [
            "Minecraft Java Programming for Beginners",
            "Build a Password generator using Python Bootcamp for kids"
        ]
    },
    {
        "query": "beginner-friendly ml course",
        "relevant_titles": [
            "From Beginner to AI Pro: Kickstart Your Journey With Artificial Intelligence! (For Kids)",
            "Machine Learning: 7-Day Project-Based Summer Camp"
        ]
    },
    {
        "query": "pro level ai course",
        "relevant_titles": [
            "AI Pro Camp: Career Growth Catalyst",
            "AI Pro: Creative Writing Camp for Adults"
        ]
    },
    {
        "query": "stock market using ai",
        "relevant_titles": [
            "AI in Stock Market Success: Career Growth Camp",
            "The Millionaire's AI Playbook: Learn How to Predict Stock Prices! (For Kids)"
        ]
    },
]


# Evaluation Metrics
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = sum([1 for title in recommended_k if title in relevant])
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = sum([1 for title in recommended_k if title in relevant])
    return hits / len(relevant) if relevant else 0

def mrr(recommended, relevant):
    for rank, title in enumerate(recommended, start=1):
        if title in relevant:
            return 1 / rank
    return 0

# Running Evaluation
def evaluate_model(test_queries, top_k=5):
    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for entry in test_queries:
        query = entry["query"]
        relevant = entry["relevant_titles"]

        # Get recommendations from the retriever
        results = get_top_k_courses(query, top_k=top_k)
        recommended_titles = [
            res.metadata.get('title', 'Unknown Course') for res in results
        ]

        # Computing metrics
        precision = precision_at_k(recommended_titles, relevant, top_k)
        recall = recall_at_k(recommended_titles, relevant, top_k)
        reciprocal_rank = mrr(recommended_titles, relevant)

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(reciprocal_rank)

        # Debugging print for each query
        print(f"\nQuery: {query}")
        print(f"Recommended: {recommended_titles}")
        print(f"Relevant: {relevant}")
        print(f"Precision@{top_k}: {precision:.2f}, Recall@{top_k}: {recall:.2f}, MRR: {reciprocal_rank:.2f}")

    print("\n--- Average Metrics ---")
    print(f"Avg Precision@{top_k}: {np.mean(precision_scores):.2f}")
    print(f"Avg Recall@{top_k}: {np.mean(recall_scores):.2f}")
    print(f"Avg MRR: {np.mean(mrr_scores):.2f}")

if __name__ == "__main__":
    evaluate_model(test_queries, top_k=5)
