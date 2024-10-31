import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time

# Set page configuration
st.set_page_config(
    page_title="T5 Transformer Testing App",
    page_icon="🤖",
    layout="wide"
)

# Initialize T5 model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_t5_response(text, model, tokenizer, max_length=50):
    try:
        # Add error handling for empty or invalid text
        if not text or not isinstance(text, str):
            return "Error: Invalid input text"
            
        input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Add device handling for GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        input_ids = input_ids.to(device)
        
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            early_stopping=True
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("T5 Transformer Testing App")
    st.write("Upload a CSV file and test T5 transformer responses")
    
    # Load model
    with st.spinner("Loading T5 model..."):
        tokenizer, model = load_model()
        
    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please refresh the page and try again.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV with error handling
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV file is empty.")
                return
                
            # Display dataframe preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            text_column = st.selectbox("Select text column for T5 input:", df.columns)
            
            # Task selection
            task_prefix = st.selectbox(
                "Select T5 task:",
                ["summarize: ", "translate English to German: ", "translate English to French: ", 
                 "translate English to Romanian: ", "question: "]
            )
            
            # Number of samples and max length settings
            col1, col2 = st.columns(2)
            with col1:
                num_samples = st.slider("Number of samples to process", 1, min(10, len(df)), 3)
            with col2:
                max_length = st.slider("Maximum output length", 10, 100, 50)
            
            # Generate button
            if st.button("Generate T5 Responses"):
                st.subheader("Generated Responses")
                
                # Add a progress bar
                progress_bar = st.progress(0)
                
                # Process selected number of samples
                for i in range(num_samples):
                    # Update progress
                    progress = (i + 1) / num_samples
                    progress_bar.progress(progress)
                    
                    text = df[text_column].iloc[i]
                    
                    # Display original text
                    st.write(f"**Original Text {i+1}:**")
                    st.write(text)
                    
                    # Generate and display response
                    with st.spinner(f"Generating response for sample {i+1}..."):
                        input_text = task_prefix + str(text)
                        response = generate_t5_response(input_text, model, tokenizer, max_length)
                        
                    st.write(f"**T5 Response {i+1}:**")
                    st.write(response)
                    st.markdown("---")
                
                # Clear progress bar
                progress_bar.empty()
            
            # Add usage tips
            with st.expander("Usage Tips"):
                st.markdown("""
                - The T5 model works best with clear, concise text
                - Different task prefixes will yield different types of outputs
                - For best results, ensure your input text is in English
                - The model has a maximum input length of 512 tokens
                - If using GPU, generation will be faster
                - Large files may take longer to process
                """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
