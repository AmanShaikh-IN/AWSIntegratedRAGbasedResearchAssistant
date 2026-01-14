# AWS Configuration
BEDROCK_REGION = "us-east-1"  # Your AWS region
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Model Configuration
MODEL_CONFIG = {
    "maxTokens": 4096,
    "temperature": 0.7,
    "topP": 0.9
}

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_K = 5

# Streamlit Configuration
PAGE_TITLE = "Research Assistant"
PAGE_ICON = "📚"
