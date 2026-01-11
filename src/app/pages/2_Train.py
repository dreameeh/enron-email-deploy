import streamlit as st
import pandas as pd
from src.models.finetune_llama import LlamaEmailTuner
import logging

logger = logging.getLogger(__name__)

class TrainingUI:
    def __init__(self):
        st.title("Train Model")
        self.tuner = LlamaEmailTuner()
        
        # Initialize session state for pre-prompts
        if 'pre_prompts' not in st.session_state:
            st.session_state.pre_prompts = [""] * 5  # 5 empty pre-prompts
        
        self.render()
    
    def render(self):
        """Render the training UI."""
        with st.form("training_form"):
            # Training parameters
            st.subheader("Training Parameters")
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=3e-4,
                format="%.0e"
            )
            
            num_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=10,
                value=1
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=8
            )
            
            # Pre-prompts
            st.subheader("Pre-prompts (Optional)")
            st.write("Add up to 5 pre-prompts that will be randomly selected during training.")
            
            pre_prompts = []
            for i in range(5):
                prompt = st.text_area(
                    f"Pre-prompt {i+1}",
                    value=st.session_state.pre_prompts[i],
                    height=100,
                    key=f"pre_prompt_{i}"
                )
                if prompt.strip():
                    pre_prompts.append(prompt.strip())
                    st.session_state.pre_prompts[i] = prompt
            
            # Training button
            train_button = st.form_submit_button("Start Training")
            
            if train_button:
                try:
                    # Load data
                    df = pd.read_parquet('data/samples/email_processed_1000.parquet')
                    
                    # Initialize tuner with parameters
                    tuner = LlamaEmailTuner(
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        pre_prompts=pre_prompts if pre_prompts else None
                    )
                    
                    # Prepare dataset
                    with st.spinner("Preparing training data..."):
                        dataset = tuner.prepare_training_data(df)
                    
                    # Train
                    with st.spinner("Training model..."):
                        version = tuner.train(dataset)
                        st.success(f"Training complete! Model saved as version: {version}")
                    
                    # Show available versions
                    versions = tuner.cache.list_versions()
                    st.write("Available model versions:")
                    for v, info in versions.items():
                        st.write(f"- Version: {v}")
                        if 'pre_prompts' in info:
                            st.write("  Pre-prompts:")
                            for i, prompt in enumerate(info['pre_prompts'], 1):
                                st.write(f"  {i}. {prompt}")
                
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    logger.error(f"Training error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    ui = TrainingUI()
