import streamlit as st
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.models.finetune_llama import LlamaEmailTuner
from src.models.model_cache import ModelCache

class ChatUI:
    def __init__(self):
        st.set_page_config(page_title="Enron Email Chat", layout="wide")
        self.cache = ModelCache()
        self.tuner = LlamaEmailTuner()
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_persona' not in st.session_state:
            st.session_state.current_persona = "phillip.allen"
        if 'model_version' not in st.session_state:
            versions = self.cache.list_versions()
            st.session_state.model_version = list(versions.keys())[0] if versions else None
        if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
            if st.session_state.model_version:
                self.load_model()
    
    def load_model(self):
        """Load the selected model version."""
        try:
            if st.session_state.model_version:
                with st.spinner("Loading model..."):
                    # Load model and tokenizer
                    model_path = self.cache.get_model_path(st.session_state.model_version)
                    st.session_state.model = self.tuner.load_model(model_path)
                    st.session_state.tokenizer = self.tuner.tokenizer
                    st.success("Model loaded successfully!")
            else:
                st.warning("No trained models available. Please train a model first.")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            
    def load_personas(self):
        """Load available personas with their details."""
        return {
            "phillip.allen": {
                "name": "Phillip Allen",
                "title": "VP of Trading",
                "style": "Direct and business-focused communication style"
            },
            "jeff.skilling": {
                "name": "Jeff Skilling",
                "title": "CEO",
                "style": "Authoritative and strategic communication"
            },
            "kenneth.lay": {
                "name": "Kenneth Lay",
                "title": "Chairman",
                "style": "Formal and corporate leadership style"
            },
            "custom": {
                "name": "Custom Persona",
                "title": "Custom Role",
                "style": "Define your own communication style"
            }
        }
        
    def render_chat_header(self):
        """Render the chat interface header with persona selection."""
        st.title("💬 Enron Email Chat")
        
        # Persona selection
        personas = self.load_personas()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_persona = st.selectbox(
                "Select Email Persona",
                options=list(personas.keys()),
                format_func=lambda x: f"{personas[x]['name']} - {personas[x]['title']}",
                key="persona_selector"
            )
            
        # Show persona details
        with col2:
            st.info(f"Communication Style:\n{personas[selected_persona]['style']}")
            
        # Custom persona settings
        if selected_persona == "custom":
            with st.expander("Custom Persona Settings"):
                st.text_input("Name", key="custom_name")
                st.text_input("Title", key="custom_title")
                st.text_area(
                    "Communication Style",
                    key="custom_style",
                    help="Describe the communication style for this persona"
                )
        
        # Update current persona if changed
        if selected_persona != st.session_state.current_persona:
            st.session_state.current_persona = selected_persona
            st.session_state.messages = []  # Clear chat history on persona change
            
    def render_chat_messages(self):
        """Render chat messages."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def generate_response(self, prompt):
        """Generate response using the selected model."""
        if not st.session_state.model_version or 'model' not in st.session_state:
            return "Error: No model loaded. Please train a model first."
            
        try:
            # Get current persona details
            personas = self.load_personas()
            current_persona = st.session_state.current_persona
            name = personas[current_persona]["name"]
            title = personas[current_persona]["title"]
            
            # Create the full prompt
            full_prompt = f"""You are {name}, {title} at Enron. Write an email response in your communication style.

Previous messages:
{chr(10).join([f"{'User' if m['role'] == 'user' else name}: {m['content']}" for m in st.session_state.messages[-5:]])}

{name}:"""
            
            # Tokenize input
            inputs = st.session_state.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate response
            temperature = st.session_state.get('temperature', 0.7)
            max_length = st.session_state.get('max_length', 200)
            
            with torch.no_grad():
                outputs = st.session_state.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=st.session_state.tokenizer.pad_token_id,
                    eos_token_id=st.session_state.tokenizer.eos_token_id
                )
            
            # Decode response
            response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"Error: Failed to generate response. Please try again."
    
    def render_chat_input(self):
        """Render chat input and handle message generation."""
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and add assistant response
            with st.chat_message("assistant"):
                response = self.generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    def render_chat_controls(self):
        """Render additional chat controls."""
        with st.sidebar:
            st.header("Chat Controls")
            
            # Clear chat
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            # Model selection
            st.subheader("Model Settings")
            versions = self.cache.list_versions()
            if versions:
                selected_version = st.selectbox(
                    "Select Model Version",
                    options=list(versions.keys()),
                    format_func=lambda x: f"{x} ({versions[x]['timestamp'][:10]})"
                )
                
                # Reload model if version changed
                if selected_version != st.session_state.model_version:
                    st.session_state.model_version = selected_version
                    self.load_model()
            else:
                st.info("No trained models available. Please train a model first.")
            
            # Temperature slider
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values make the output more random"
            )
            st.session_state.temperature = temperature
            
            # Max length slider
            max_length = st.slider(
                "Max Response Length",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Maximum number of tokens in the response"
            )
            st.session_state.max_length = max_length
    
    def main(self):
        """Main UI rendering loop."""
        self.render_chat_header()
        self.render_chat_messages()
        self.render_chat_input()
        self.render_chat_controls()

if __name__ == "__main__":
    ui = ChatUI()
    ui.main()
