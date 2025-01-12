# #using faiss - the real one
from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load Bloom model and tokenizer (using CPU only)
model_name = "bigscience/bloomz-560m"  # Lightweight Bloom model
bloom_model = AutoModelForCausalLM.from_pretrained(model_name).cpu()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example scripture data for Sentence Transformer
scripture_data = [
    {"sanskrit": "तपः स्वाध्यायेश्वरप्रणिधानानि क्रियायोगः।", "translation": "Discipline, self-study, and surrender to the divine constitute yoga practice."},
    {"sanskrit": "योगश्चित्तवृत्तिनिरोधः।", "translation": "Yoga is the cessation of the fluctuations of the mind."},
    {"sanskrit": "अभ्यासवैराग्याभ्यां तन्निरोधः।", "translation": "Through practice and detachment, the mind is controlled."},
    {"sanskrit": "सत्त्वं रजस्तम इति गुणाः प्रकृतिसंभवाः।", "translation": "The qualities of nature are purity, passion, and inertia."},
    {"sanskrit": "ध्यानजं ज्ञानम।", "translation": "Knowledge is born of meditation."}
]

# Pre-compute embeddings for scriptures using Sentence Transformer
scripture_embeddings = np.array([sentence_model.encode(verse['translation'], convert_to_tensor=True) for verse in scripture_data])

# Load embeddings and metadata for Bloom-based retrieval
gita_embeddings = np.load('./data/gita_embeddings.npy')
pys_embeddings = np.load('./data/pys_embeddings.npy')
gita_verses = np.load('./data/gita_verses.npy', allow_pickle=True)
pys_verses = np.load('./data/pys_verses.npy', allow_pickle=True)

# FAISS setup: Creating an index for fast similarity search
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # The number of dimensions of your embeddings
    index = faiss.IndexFlatL2(dimension)  # Use L2 (Euclidean) distance for similarity
    index.add(embeddings)  # Adding the embeddings to the FAISS index
    return index

# Create FAISS indices for scripture, Gita, and PYS embeddings
scripture_index = create_faiss_index(scripture_embeddings)
gita_index = create_faiss_index(gita_embeddings)
pys_index = create_faiss_index(pys_embeddings)

# Function to retrieve top results using FAISS index
def retrieve_top_results(query_embedding, index, verses, top_n=4):
    # Perform a similarity search with FAISS
    distances, indices = index.search(np.array([query_embedding]), top_n)
    top_results = [verses[i] for i in indices[0]]
    return top_results

# Function to generate summary using Bloom-based model
def generate_summary(verses, query):
    # if source == "gita":
    #     context = "\n".join([f"{v[3]} (Translation: {v[4]})" for v in verses])
    # elif source == "pys":
    #     context = "\n".join([f"{v[2]} (Translation: {v[3]})" for v in verses])

    context = "\n".join([f"{v[3]} (Translation: {v[4]})" for v in verses])
    prompt = f"User Query: {query}\n\nRelevant Verses:\n{context}\n\nSummary:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = bloom_model.generate(**inputs, max_length=500, num_return_sequences=1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        # Get user input from the request
        data = request.json
        query = data.get('query')
        source = data.get('source')

        if not query or not source:
            return jsonify({"error": "Query and source are required"}), 400

        # Encode the query using the SentenceTransformer model
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)

        if source == "gita":
            results = retrieve_top_results(query_embedding, gita_index, gita_verses)
            summary = generate_summary(results, query)
            response = {
                "results": [{"sanskrit": r[3], "translation": r[4]} for r in results],
                "summary": summary
            }

        elif source == "pys":
            results = retrieve_top_results(query_embedding, pys_index, pys_verses)
            summary = generate_summary(results, query)
            response = {
                "results": [{"sanskrit": r[2], "translation": r[3]} for r in results],
                "summary": summary
            }

        elif source == "sentence":
            results = retrieve_top_results(query_embedding, scripture_index, scripture_data)
            response = {
                "user_query": query,
                "relevant_verses": results
            }

        else:
            return jsonify({"error": "Invalid source"}), 400

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, use_reloader=False)