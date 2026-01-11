import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.finetune_llama import LlamaEmailTuner
from src.models.model_cache import ModelCache

class ModelManagerUI:
    def __init__(self):
        st.set_page_config(page_title="Enron Email Model Manager", layout="wide")
        self.cache = ModelCache()
        self.tuner = LlamaEmailTuner()
        
    def render_sidebar(self):
        """Render the sidebar with model selection and actions."""
        with st.sidebar:
            st.title("Model Manager")
            
            # Model Selection
            st.header("Select Model")
            versions = self.cache.list_versions()
            selected_version = st.selectbox(
                "Model Version",
                options=list(versions.keys()) + ["Create New"],
                format_func=lambda x: f"{x} ({versions[x]['timestamp'][:10]})" if x in versions else x
            )
            
            # Actions
            st.header("Actions")
            action = st.radio(
                "Choose Action",
                ["Analyze", "Tune", "Delete", "Train New"]
            )
            
            return selected_version, action
    
    def render_model_metrics(self, version):
        """Display model metrics and health indicators."""
        st.header("Model Health Metrics")
        
        # Get model training args
        training_args = self.cache.get_training_args(version)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Training Loss",
                f"{training_args.get('final_loss', 0.0):.4f}",
                delta="-0.123"
            )
        
        with col2:
            st.metric(
                "Perplexity",
                f"{training_args.get('perplexity', 0.0):.2f}",
                delta="-1.2"
            )
        
        with col3:
            st.metric(
                "Response Time",
                f"{training_args.get('avg_response_time_ms', 0):.0f}ms",
                delta="-50ms"
            )
        
        # Training Progress
        st.subheader("Training Progress")
        if training_args and 'loss_history' in training_args:
            df = pd.DataFrame(training_args['loss_history'])
            fig = px.line(df, x='step', y='loss', title='Training Loss Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_confidence_analysis(self, version):
        """Show confidence analysis and distribution."""
        st.header("Confidence Analysis")
        
        # Sample confidence scores
        scores = pd.DataFrame({
            'confidence': [0.92, 0.85, 0.78, 0.95, 0.88, 0.72],
            'category': ['Business', 'Personal', 'Technical', 'Meeting', 'Report', 'Other']
        })
        
        # Confidence Distribution
        fig = go.Figure(data=[
            go.Bar(x=scores['category'], y=scores['confidence'], name='Confidence')
        ])
        fig.update_layout(title='Confidence by Email Category')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_adjustment(self, version):
        """Interface for adjusting model parameters."""
        st.header("Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Parameters")
            learning_rate = st.slider(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-3,
                value=2e-4,
                format="%.0e"
            )
            
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=4
            )
            
            num_epochs = st.slider(
                "Number of Epochs",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with col2:
            st.subheader("LoRA Parameters")
            lora_r = st.slider(
                "LoRA Rank",
                min_value=1,
                max_value=64,
                value=8
            )
            
            lora_alpha = st.slider(
                "LoRA Alpha",
                min_value=1,
                max_value=64,
                value=32
            )
        
        if st.button("Apply Changes"):
            st.info("Applying parameter changes...")
            # TODO: Implement parameter update logic
    
    def render_inbox_selection(self):
        """Interface for selecting email inbox/persona."""
        st.header("Email Persona Selection")
        
        # Load available personas
        personas = {
            "phillip.allen": "Phillip Allen - VP of Trading",
            "jeff.skilling": "Jeff Skilling - CEO",
            "kenneth.lay": "Kenneth Lay - Chairman",
            "custom": "Custom Persona"
        }
        
        selected_persona = st.selectbox(
            "Select Email Persona",
            options=list(personas.keys()),
            format_func=lambda x: personas[x]
        )
        
        if selected_persona == "custom":
            st.text_input("Name")
            st.text_input("Title")
            st.text_area("Description of communication style")
            st.file_uploader("Upload sample emails (optional)")
    
    def render_training_interface(self):
        """Interface for training a new model."""
        st.header("Train New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Selection")
            st.file_uploader("Upload Training Data", type=["csv", "parquet"])
            st.number_input("Number of Training Examples", min_value=100, value=1000)
            
        with col2:
            st.subheader("Training Configuration")
            st.selectbox("Base Model", ["TinyLlama-1.1B", "Llama-2-7B"])
            st.checkbox("Enable Early Stopping")
            st.checkbox("Save Checkpoints")
        
        if st.button("Start Training"):
            st.info("Starting training process...")
            # TODO: Implement training start logic
    
    def render_delete_interface(self, version):
        """Interface for deleting models."""
        st.header("Delete Model")
        
        st.warning(
            f"Are you sure you want to delete model version {version}? "
            "This action cannot be undone."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            confirm = st.text_input(
                "Type 'DELETE' to confirm",
                key="delete_confirm"
            )
        
        with col2:
            if st.button("Delete Model") and confirm == "DELETE":
                st.error("Deleting model...")
                # TODO: Implement model deletion logic
    
    def main(self):
        """Main UI rendering loop."""
        st.sidebar.title("Navigation")
        st.sidebar.write("Use the pages in the sidebar to:")
        st.sidebar.write("1. **Chat**: Chat with the trained model")
        st.sidebar.write("2. **Train**: Train a new model with custom pre-prompts")
        
        selected_version, action = self.render_sidebar()
        
        if action == "Analyze" and selected_version != "Create New":
            self.render_model_metrics(selected_version)
            self.render_confidence_analysis(selected_version)
            
        elif action == "Tune" and selected_version != "Create New":
            self.render_parameter_adjustment(selected_version)
            
        elif action == "Delete" and selected_version != "Create New":
            self.render_delete_interface(selected_version)
            
        elif action == "Train New" or selected_version == "Create New":
            self.render_training_interface()
        
        # Always show inbox selection
        self.render_inbox_selection()

if __name__ == "__main__":
    ui = ModelManagerUI()
    ui.main()
