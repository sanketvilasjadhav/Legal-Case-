import os
import random
import warnings
import json
import re
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA MANAGEMENT & MOCK GENERATOR
# ==========================================
class DataManager:
    """
    Handles data loading. If files don't exist, generates robust mock data
    so the app runs immediately without external dependencies.
    """
    def __init__(self):
        self.case_data_path = "case_texts_with_citations.csv"
        self.graph_data_path = "legal_graph_data.csv"
        self.ensure_data_exists()

    def ensure_data_exists(self):
        if not os.path.exists(self.case_data_path):
            print("⚠️ Data files not found. Generating synthetic legal data...")
            self._generate_synthetic_cases()
        if not os.path.exists(self.graph_data_path):
            self._generate_synthetic_graph_data()
            
    def _generate_synthetic_cases(self):
        # Create dummy cases for retrieval (AJCL)
        data = {
            'case_id': range(100),
            'case_text': [
                f"The plaintiff argues a breach of contract under Section {random.randint(10,99)}. " 
                f"The defendant cites precedent {random.randint(1990, 2020)} regarding negligence."
                for _ in range(100)
            ],
            'outcome': [random.choice(['Granted', 'Dismissed']) for _ in range(100)],
            'court_level': [random.choice(['District', 'Appellate', 'Supreme']) for _ in range(100)]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.case_data_path, index=False)
        print(f"✅ Generated {self.case_data_path}")

    def _generate_synthetic_graph_data(self):
        # Create dummy entities for CLR-Graph
        entities = ['Plaintiff', 'Defendant', 'Contract', 'Tort', 'Negligence', 'Damages', 'Section 54', 'Precedent A']
        edges = []
        for _ in range(50):
            src = random.choice(entities)
            dst = random.choice(entities)
            if src != dst:
                edges.append((src, dst))
        
        df = pd.DataFrame(edges, columns=['source', 'target'])
        df.to_csv(self.graph_data_path, index=False)
        print(f"✅ Generated {self.graph_data_path}")

    def load_cases(self):
        return pd.read_csv(self.case_data_path)

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

class LightweightGCN(nn.Module):
    """
    A pure PyTorch implementation of GCN to avoid torch_geometric dependency issues.
    Used for CLR-Graph (Case Law Reasoning).
    """
    def __init__(self, num_features, hidden_dim, output_dim):
        super(LightweightGCN, self).__init__()
        self.linear1 = nn.Linear(num_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # Layer 1: A * X * W1
        x = torch.matmul(adj, x)
        x = self.linear1(x)
        x = self.relu(x)
        
        # Layer 2: A * X * W2
        x = torch.matmul(adj, x)
        x = self.linear2(x)
        return x

class LegalAI_Engine:
    """
    The Core Brain. Initializes all sub-models (HiLPE, MLOE, etc.)
    """
    def __init__(self):
        print("🚀 Initializing Legal AI Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- HiLPE & AJCL Models (BERT) ---
        print("   -> Loading BERT (HiLPE/AJCL)...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.classifier = nn.Linear(768, 2).to(self.device) # Binary outcome
        
        # --- MLOE Model (T5 for Explanation) ---
        print("   -> Loading T5 (MLOE)...")
        self.gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        
        # --- CLR Graph Model ---
        print("   -> Initializing CLR Graph Network...")
        self.gcn = LightweightGCN(num_features=64, hidden_dim=32, output_dim=16).to(self.device)
        
        # Load Knowledge Base
        self.data_manager = DataManager()
        self.knowledge_base = self.data_manager.load_cases()
        
        # Pre-compute KB embeddings for search
        self.kb_embeddings = self._precompute_kb_embeddings()

    def _precompute_kb_embeddings(self):
        # Create simple random embeddings for the mock KB to save startup time
        # In production, run real BERT over self.knowledge_base['case_text']
        return np.random.rand(len(self.knowledge_base), 768)

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy() # CLS token

    def predict_outcome(self, text):
        """HiLPE-Net Logic: Predict outcome & Confidence"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            # Extract features
            outputs = self.bert_model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            # Pass through classifier
            logits = self.classifier(cls_token)
            probs = F.softmax(logits, dim=1)
            
        confidence, pred_class = torch.max(probs, 1)
        label = "GRANTED" if pred_class.item() == 1 else "DISMISSED"
        return label, confidence.item()

    def generate_explanation(self, text, outcome):
        """MLOE-Net Logic: Generate justification"""
        input_text = f"explain legal outcome {outcome}: {text}"
        inputs = self.gen_tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(self.device)
        
        outputs = self.gen_model.generate(
            **inputs, 
            max_length=100, 
            num_beams=4, 
            no_repeat_ngram_size=2
        )
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def analyze_graph_entities(self, text):
        """CLR-Graph Logic: Extract entities and build a graph"""
        # 1. Simple Regex Entity Extraction (Simulated NER)
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        if not entities:
            entities = {"Plaintiff", "Defendant", "Court", "Evidence"}
            
        entities = list(entities)[:10] # Limit size
        
        # 2. Build Adjacency Matrix (Simulating relations)
        size = len(entities)
        adj = torch.eye(size).to(self.device) # Self-loops
        
        # Add random edges for demonstration of reasoning paths
        edges = []
        for i in range(size):
            for j in range(i+1, size):
                if random.random() > 0.7:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edges.append((entities[i], entities[j]))
                    
        # 3. Run GCN
        # Fake features for nodes
        features = torch.randn(size, 64).to(self.device)
        with torch.no_grad():
            embeddings = self.gcn(features, adj)
            
        return entities, edges, embeddings.cpu().numpy()

    def find_similar_precedents(self, text):
        """AJCL-Net Logic: Contrastive Search"""
        # Get query embedding
        query_vec = self.get_bert_embedding(text)
        
        # Cosine similarity with KB
        sims = cosine_similarity(query_vec, self.kb_embeddings).flatten()
        
        # Get top 3
        top_indices = sims.argsort()[-3:][::-1]
        results = []
        for idx in top_indices:
            row = self.knowledge_base.iloc[idx]
            results.append({
                "case_id": row['case_id'],
                "text": row['case_text'][:100] + "...",
                "outcome": row['outcome'],
                "score": float(sims[idx])
            })
        return results

    def compute_gradients(self, text):
        """OSGA-Net Logic: Compute gradient sensitivity for heatmap"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        
        # Enable embeddings gradient
        embeddings = self.bert_model.get_input_embeddings()
        inputs_embeds = embeddings(inputs['input_ids'])
        inputs_embeds.retain_grad()
        
        # Forward pass
        outputs = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        # Backward pass on top prediction
        target_class = logits.argmax()
        self.bert_model.zero_grad()
        logits[0, target_class].backward()
        
        # Get gradients
        grads = inputs_embeds.grad[0].cpu().abs().sum(dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Normalize
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
        
        return tokens, grads.numpy()

# Initialize Engine Global
engine = LegalAI_Engine()

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================

def plot_graph(entities, edges):
    """Generates a NetworkX graph image"""
    plt.figure(figsize=(8, 6))
    G = nx.Graph()
    G.add_nodes_from(entities)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
    plt.title("CLR-Graph: Entity Reasoning Network")
    
    # Save to buffer
    fig_path = "graph_output.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path

def plot_attention_heatmap(tokens, scores):
    """Generates a bar chart of token importance"""
    # Filter special tokens
    filtered = [(t, s) for t, s in zip(tokens, scores) if t not in ['[CLS]', '[SEP]', '[PAD]']]
    labels, values = zip(*filtered[:15]) # Top 15 for readability
    
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(labels), y=list(values), palette="viridis")
    plt.xticks(rotation=45)
    plt.title("OSGA-Net: Token Importance (Sensitivity)")
    
    fig_path = "heatmap_output.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    return fig_path

# ==========================================
# 4. GRADIO INTERFACE LOGIC
# ==========================================

def pipeline_wrapper(legal_text):
    if not legal_text.strip():
        return "Please enter text.", "", "", None, None, ""

    # 1. Prediction (HiLPE)
    outcome, confidence = engine.predict_outcome(legal_text)
    outcome_str = f"⚖️ **{outcome}**\n(Confidence: {confidence:.2%})"

    # 2. Explanation (MLOE)
    explanation = engine.generate_explanation(legal_text, outcome)

    # 3. Similar Cases (AJCL)
    precedents = engine.find_similar_precedents(legal_text)
    prec_str = "📚 **Relevant Precedents found in KB:**\n"
    for p in precedents:
        prec_str += f"- [Case {p['case_id']}] ({p['outcome']}) - Sim: {p['score']:.2f}\n"

    # 4. Graph Reasoning (CLR)
    entities, edges, _ = engine.analyze_graph_entities(legal_text)
    graph_img = plot_graph(entities, edges)

    # 5. Gradient Analysis (OSGA)
    tokens, grads = engine.compute_gradients(legal_text)
    heatmap_img = plot_attention_heatmap(tokens, grads)

    return outcome_str, explanation, prec_str, graph_img, heatmap_img, "✅ Analysis Complete"

# ==========================================
# 5. UI LAYOUT
# ==========================================

custom_css = """
.gradio-container {background-color: #f0f2f6}
.result-box {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1)}
"""

with gr.Blocks(css=custom_css, title="Legal AI Decision Support") as demo:
    gr.Markdown("# ⚖️ Advanced Legal AI System (HiLPE + OSGA + MLOE + CLR + AJCL)")
    gr.Markdown("Input case facts below to generate outcome predictions, reasoning graphs, and precedent analysis.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=8, 
                placeholder="e.g., The plaintiff suffered damages due to the defendant's negligence in maintaining the property under Section 45...",
                label="Legal Case Text"
            )
            analyze_btn = gr.Button("🚀 Run Legal Analysis", variant="primary")
        
        with gr.Column(scale=1):
            outcome_box = gr.Markdown(label="Predicted Outcome", value="Waiting for input...")
            status_box = gr.Textbox(label="System Status", interactive=False)

    with gr.Tabs():
        with gr.TabItem("📝 Explanation & Precedents"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🤖 Generated Justification (MLOE-Net)")
                    explain_box = gr.Textbox(lines=4, label="AI Reasoning")
                with gr.Column():
                    gr.Markdown("### 📚 Similar Precedents (AJCL-Net)")
                    precedent_box = gr.Markdown()
        
        with gr.TabItem("🕸️ Reasoning Graph (CLR)"):
            gr.Markdown("Visualizing entity relationships extracted from the text.")
            graph_output = gr.Image(label="Entity Graph")
            
        with gr.TabItem("🔍 Attention Heatmap (OSGA)"):
            gr.Markdown("Which words influenced the AI's decision the most?")
            heatmap_output = gr.Image(label="Gradient Sensitivity")

    analyze_btn.click(
        pipeline_wrapper,
        inputs=[input_text],
        outputs=[outcome_box, explain_box, precedent_box, graph_output, heatmap_output, status_box]
    )

if __name__ == "__main__":
    print("Starting Web Server...")
    demo.launch(share=True)
