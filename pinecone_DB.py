import os
import pickle
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import json

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initializng Pinecone     &     Environment: pinecone runs on different platforms ; us-east-1-aws is US East-1 region Amazon Web Services.
pc = Pinecone(api_key=pinecone_api_key)

# Creating an index : which is an identifier and is a database table where our embeddings are stored in pinecone.
index_name = "course-embeddings"

if index_name in pc.list_indexes().names():
    print(f"{index_name} was found and trying to delete that index")
    pc.delete_index(index_name)
    print("Deleted existing index successfully")

pc.create_index(
    name=index_name,
    dimension=768, 
    metric="cosine", 
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Connecting to the index
index = pc.Index(index_name)
print("Pinecone initialized and index created successfully!")

with open("batched_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

with open("courses.json", "r", encoding="utf-8") as f:   
    data = json.load(f)

# Converting the embeddings into required format(ID, vector, metadata)  &  Uploadings vectors to Pinecone.
def detect_level(text):
    text = text.lower()
    if "beginner" in text or "introduction" in text:
        return "beginner"
    elif "intermediate" in text:
        return "intermediate"
    elif "advanced" in text or "pro" in text:
        return "advanced"
    else:
        return "unspecified"

vectors = []
for i, embedding in enumerate(embeddings):
    course_info = data[i]  # As data is a list of course dictionaries
    title = (course_info.get("Title") or "Unknown Course").strip()
    description = (course_info.get("Description") or "No description available").strip()

    metadata = {
    "title": title,
    "description": description,
    "text": f"{title}. {description}",
    "details_link": course_info.get("More_Details_Link", ""),
    "curriculum_link": course_info.get("Curriculum_Link", ""),
    "course_fee_link": course_info.get("Course_Fee_Link", ""),
    "enquire_link": course_info.get("Enquire_Link", ""),
    "demo_link": course_info.get("Demo_Booking_Link", ""),
    "level": detect_level(title + " " + description),
    "search_count": 0
    }

    vectors.append((str(i), embedding.tolist(), metadata))    # Id, Vector, Metadata 

# Uploading embeddings with metadata
index.upsert(vectors)
print("Embeddings and metadata uploaded successfully!")
