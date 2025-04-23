import os
import re
import string
import nltk
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import pickle 

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define index
index_name = "course-embeddings"
index = pc.Index(index_name)
model = SentenceTransformer("all-mpnet-base-v2")

nltk.download("stopwords")

synonym_dict = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "ds": "data science",
    "dl": "deep learning",
    "cv": "computer vision",
    "web dev": "web development",
    "js": "javascript",
    "proj": "project",
    "intro": "introduction",
    "bootcamp": "course",
    "course": "bootcamp"
}

# Query preprocessing
def preprocess_query(query):
    query = query.lower()
    for key, value in synonym_dict.items():
        query = query.replace(key, value)
    query = re.sub(rf"[{re.escape(string.punctuation)}]", "", query)
    tokens = query.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def extract_topic_keywords(query):
    synonyms = {
        "kids": ["kids", "children", "junior", "school", "students", "young minds", "youth", "beginners"],
        "ai": ["ai", "artificial intelligence", "machine learning", "ml"],
        "web": ["web", "website", "frontend", "html", "css", "javascript", "web development"],
        "python": ["python", "py"],
        "cloud": ["cloud", "aws", "azure", "gcp", "cloud computing"],
        "beginner": ["beginner", "intro", "introduction", "getting started", "no experience"],
        "game": ["game", "game dev", "unity", "unreal", "game development"],
        "learn java programming": ["core java", "java course", "java coding", "object oriented programming"],
        "pro": ["advanced", "expert", "intermediate"]
    }

    # Clean and tokenize
    query = query.lower()
    query = re.sub(rf"[{re.escape(string.punctuation)}]", "", query)
    tokens = query.split()
    stop_words = set(stopwords.words("english"))
    keywords = [word for word in tokens if word not in stop_words]

    # Expand keywords using synonym map
    expanded_keywords = set()
    for word in keywords:
        for syn_group in synonyms.values():
            if word in syn_group:
                expanded_keywords.update(syn_group)
                break
        else:
            expanded_keywords.add(word)
    return list(expanded_keywords)

def keyword_relevance(metadata, keywords):
    text = (metadata.get("title", "") + " " + metadata.get("description", "")).lower()
    return sum(1 for kw in keywords if kw in text) / len(keywords)

def get_level_from_query(query):
    query = query.lower()
    if "beginner" in query or "introduction" in query:
        return "beginner"
    elif "intermediate" in query:
        return "intermediate"
    elif "advanced" in query or "pro" in query:
        return "advanced"
    return None

def reranked_results(query, top_k=5):
    level_filter = get_level_from_query(query)

    # Generating embedding for the query using Sentence Transformers
    preprocessed_query = preprocess_query(query)
    query_embedding = model.encode([preprocessed_query])[0].tolist()

    pinecone_filter = {"level": level_filter} if level_filter else {}
    results = index.query(vector=[query_embedding], top_k=top_k*6, include_metadata=True, filter=pinecone_filter)

    if not results.matches:
        return []

    # Extracting topic keywords and filter
    topic_keywords = extract_topic_keywords(query)
    def is_strongly_relevant(metadata):
        text = (metadata.get("title", "") + " " + metadata.get("description", "")).lower()
        return sum(1 for kw in topic_keywords if kw in text) >= 2 

    candidate_results = [m for m in results.matches if is_strongly_relevant(m.metadata)]
    candidate_results = candidate_results if candidate_results else results.matches

    # Getting min and max search count to normalize
    search_counts = [m.metadata.get("search_count", 0) for m in candidate_results]
    min_count = min(search_counts)
    max_count = max(search_counts)
    range_count = max_count - min_count if max_count != min_count else 1

    # Normalize search count and combine with similarity score
    for match in candidate_results:
        count = match.metadata.get("search_count", 0)
        normalized_popularity = (count - min_count) / range_count
        keyword_score = keyword_relevance(match.metadata, topic_keywords)

        # New hybrid score formula: 60% semantic + 20% popularity + 20% relevance
        match.hybrid_score = 0.6 * match.score + 0.2 * normalized_popularity + 0.2 * keyword_score

    sorted_results = sorted(candidate_results, key=lambda x: x.hybrid_score, reverse=True)
    return sorted_results[:top_k]

def get_top_k_courses(query, top_k=5):
    return reranked_results(query, top_k=top_k)

def search_courses(query, top_k=5):
    results = reranked_results(query, top_k)
    if not results:
        print("No results found.")
        return
    print(f"**Results for:** {query}")
    for match in results:
        meta = match.metadata
        print(f"\nCourse: {meta.get('title', 'Unknown Course')} (Score: {match.score:.2f})\n")
        print(f"Description: {meta.get('description', 'No description')}")
        print(f"\nCurriculum: {meta.get('curriculum_link', '')}")
        print(f"Course Fee: {meta.get('course_fee_link', '')}")
        print(f"Enquire Now: {meta.get('enquire_link', '')}")
        print(f"Demo: {meta.get('demo_link', '')}\n")
        print("-" * 50)

        # Updating search count
        course_id = match.id
        old_count = match.metadata.get("search_count", 0)
        updated_metadata = {"search_count": old_count + 1}
        index.update(id=course_id, set_metadata=updated_metadata)

# Example query
user_query = "want to learn web development"
search_courses(user_query)   

def get_top_k_courses(query, top_k=5):
    """
    External access point to use the trained hybrid model from evaluation script.
    Returns top_k recommended course titles as list of dicts.
    """
    results = reranked_results(query, top_k=top_k)
    return results 