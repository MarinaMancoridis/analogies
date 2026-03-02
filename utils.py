import os
import json
import csv
import random
from openai import OpenAI
from together import Together
from anthropic import Anthropic
import google.generativeai as genai
import re
import time
from google.api_core.exceptions import ResourceExhausted
import unicodedata
from analogies.constants import models_to_developer, models, api_keys, BENCHMARK_PATHS, FINAL_TAG

# calls apis to extract responses to a prompt 
def generate_inference(prompt, model):
    developer = models_to_developer[model]

    # -------------------------
    # OPENAI (fixed properly)
    # -------------------------
    if developer == "openai":
        client = OpenAI(api_key=api_keys["openai"])

        response = client.responses.create(
            model=model,
            input=prompt
        )

        return response.output_text.strip()

    # -------------------------
    # TOGETHER
    # -------------------------
    elif developer == "together":
        client = Together(api_key=api_keys["together"])

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        )

        return resp.choices[0].message.content.strip()

    # -------------------------
    # GEMINI
    # -------------------------
    elif developer == "gemini":
        genai.configure(api_key=api_keys["gemini"])
        gem_model = genai.GenerativeModel(model)

        backoff = 1
        while True:
            try:
                return gem_model.generate_content(prompt).text.strip()
            except ResourceExhausted as e:
                delay = getattr(e, "retry_delay", None)
                delay = delay.seconds if delay else backoff
                print(f"[Gemini] rate-limited, sleeping {delay}s")
                time.sleep(delay)
                backoff = min(backoff * 2, 60)

    # -------------------------
    # CLAUDE
    # -------------------------
    elif developer == "claude":
        client = Anthropic(api_key=api_keys["claude"])

        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return message.content[0].text.strip()

    # -------------------------
    # OPENROUTER
    # -------------------------
    elif models_to_developer[model] == "openrouter":
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_keys["openrouter"],
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [{
                    "type": "text",
                    "text": prompt
                }]}],
            )   
            return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown developer for model: {model}")

def save_result(save_dir, result_dict):
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(".json")]
    next_id = len(existing_files)
    save_path = os.path.join(save_dir, f"{next_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
