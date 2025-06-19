import streamlit as st
from model import GPTNeoModel

# Set Streamlit app title
st.title("GPT-Neo Text Generation")

# Sidebar configuration
st.sidebar.header("Model Settings")
model_name = st.sidebar.text_input(
    "Model Name or Path",
    value="EleutherAI/gpt-neo-1.3B"
)
max_length = st.sidebar.slider(
    "Max Length", min_value=10, max_value=1024, value=100, step=10
)
temperature = st.sidebar.slider(
    "Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.05
)
top_p = st.sidebar.slider(
    "Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95, step=0.01
)
top_k = st.sidebar.slider(
    "Top-k", min_value=0, max_value=100, value=50, step=1
)
num_return_sequences = st.sidebar.slider(
    "Number of Sequences", min_value=1, max_value=5, value=1, step=1
)
stop_token = st.sidebar.text_input(
    "Stop Token (optional)", value=""
)

# Prompt input
prompt = st.text_area("Enter your prompt:", value="Once upon a time,")

# Button for text generation
if st.button("Generate"):
    with st.spinner("Loading model and generating text..."):
        # Load and cache model to avoid re-loading for every run
        @st.cache_resource(show_spinner=False)
        def load_model(model_name):
            return GPTNeoModel(model_name_or_path=model_name)
        gpt_neo = load_model(model_name)
        
        # Generate output
        output = gpt_neo.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            stop_token=stop_token if stop_token else None
        )
        
        # Display output(s)
        if num_return_sequences > 1:
            st.subheader("Generated Sequences")
            for i, seq in enumerate(output, 1):
                st.markdown(f"**Sequence {i}:**\n\n{seq}")
        else:
            st.subheader("Generated Text")
            st.write(output)