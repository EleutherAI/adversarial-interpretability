import os
import json
import logging
import time
from dotenv import load_dotenv
import requests
import random
import string

load_dotenv()  # Loads .env if present
DEFAULT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "openai/gpt-4o-mini")

logger = logging.getLogger(__name__)

def generate(
    prompt_text,    
    system_text = "You are a Diplomacy AI. Play faithfully to your personality profile. Always try to move the game forward per your objectives. Avoid repetition in your journal. Return valid JSON.",
    model_name=DEFAULT_MODEL,
    temperature=0.0,
    retries=10  # Number of retries
):
    """
    Calls OpenAI's chat model to get Diplomacy orders using requests.
    The prompt_text should be structured JSON describing the game state.
    Retries the call up to `retries` times with a 5-second delay between attempts on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    include_seed = True
    if include_seed:
        # Generate 5 lines of 80 random alphanumeric characters
        
        seed_lines = [
            ''.join(random.choices(string.ascii_letters + string.digits, k=80))            
            for _ in range(2)
        ]
        
        #seed_lines = generate_random_number_lines(num_lines=100, num_digits=80, frequency=frequency_config)
        random_seed_block = (
            "<RANDOM SEED PLEASE IGNORE>\n" +
            "\n".join(seed_lines) +
            "\n</RANDOM SEED>"
        )
        
        system_text += '\n\n' + random_seed_block
    
    messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text}
        ]
    
    #print(prompt_text)
    if model_name == "google/gemini-2.0-flash-001":
        temperature = 0.0    

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        #"min_p": 0.1,
        "top_k": 3,
        "max_tokens": 16000,
        "response_format": {"type": "json_object"}
    }

    if True and model_name == 'deepseek/deepseek-r1':
        print('adding deepseek stuff')
        payload['provider'] = {
                    "order": [
                        #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b, r1
                        #"Mistral" # mistral-small-3
                        #"DeepSeek", # r1
                        #"Lambda", # llama-3.1-8b
                        #"NovitaAI",  # qwen-72b, llama-3.1-8b
                        #"nebius", # qwen-72b, r1
                        #"Hyperbolic", # qwen-72b
                        #"inference.net", # llama-3.1-8b
                        #"friendly", # r1
                        #"fireworks", # r1
                        #"klusterai", # r1
                        "together", # r1
                    ],
                    "allow_fallbacks": True
                }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            llm_response = response.json()["choices"][0]["message"]["content"]
            #logger.debug(f"LLM raw response: {llm_response}")
            #print(llm_response)
            return llm_response

        except requests.RequestException as e:
            logger.error(f"API request failed on attempt {attempt}: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse response on attempt {attempt}: {e}")

        if attempt < retries:
            time.sleep(3)  # wait 5 seconds before retrying

    # Fallback after all retries have been exhausted
    #return {"journal_update": ["(Error calling LLM, fallback)"], "orders": []}
    return ''
