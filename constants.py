import os

# short versions of model names
models_to_short_name = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3",
    "gpt-4o": "GPT-4o",
    "gpt-4.5-preview": "GPT-4.5",
    "o3-mini": "O3-Mini",
    "o1-mini": "O1-Mini",
    "gemini-2.0-flash-exp": "Gemini-2.0",
    "claude-3-5-sonnet-20241022": "Claude-Sonnet",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-Instruct",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1",
    "Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL",
    "o1": "o1",
    "o3": "o3",
    "gpt-5": "GPT-5"
}

# mapping of models to their developers
models_to_developer = {"meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "gpt-4o": "openai",
    "gpt-4.5-preview": "openai",
    "o3-mini": "openai",
    "o1-mini": "openai",
    "gpt-5.1": "openai",
    "gpt-5.2": "openai",
    "gemini-2.0-flash-exp": "gemini",
    "claude-3-5-sonnet-20241022": "claude",
    "mistralai/Mistral-7B-Instruct-v0.2": "together",
    "deepseek-ai/DeepSeek-V3": "together",
    "deepseek-ai/DeepSeek-R1": "together",
    "Qwen/Qwen2-VL-72B-Instruct": "together",
    "Qwen/Qwen2-72B-Instruct": "together",
    "o1": "openai",
    "o3": "openai",
    "gpt-5": "openai",
    "google/gemini-2.5-flash": "openrouter",
    "anthropic/claude-sonnet-4.5": "openrouter",
    "anthropic/claude-opus-4.5": "openrouter",
    "deepseek/deepseek-chat": "openrouter",
    "google/gemini-3-flash-preview": "openrouter",
    "google/gemini-3.1-pro-preview": "openrouter",
    "mistralai/mistral-large": "openrouter",
    "x-ai/grok-4": "openrouter",
    "x-ai/grok-4.1-fast": "openrouter",
    "gpt-5-pro": "openai"
}
models = list(models_to_developer.keys())

# API keys for different providers
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY")
}

# paths to benchmarks
BENCHMARK_PATHS = {
    'bbh': './benchmarks/bbh',
    'mmlu': './benchmarks/mmlu'
}

# final tag
FINAL_TAG = "FINAL ANSWER:"        