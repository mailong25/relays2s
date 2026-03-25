from litellm import completion, acompletion, batch_completion, embedding
import asyncio
import yaml
from typing import Type, Any, Dict, Tuple, List, Optional
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
import os
import re
import uuid
import time
import json
from tqdm import tqdm
from openai import OpenAI
from google import genai
from google.genai import types

_env = Environment(undefined=StrictUndefined, autoescape=False)
def build_prompt(
    prompts_path : str, 
    prompt_key: str, 
    **kwargs: Any,
) -> Tuple[str, Dict[str, str]]:
    """
    Loads a single prompt template and params from YAML, renders the template, 
    and returns a dictionary containing the rendered 'prompt' string and 'params'.
    Args:
        prompts_path (str): Path to the YAML file containing prompt templates.
        prompt_key (str): Key identifying which prompt template to use.
        **kwargs: Variables to render into the prompt template.
    Returns:
        Tuple[str, Dict[str, str]]: Rendered prompt string and additional parameters.
    """
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f) or {}
    
    if prompt_key not in prompts:
        raise KeyError(f"Prompt '{prompt_key}' not found in {prompts_path}")
    prompt_config = prompts[prompt_key]

    template_string = prompt_config.get("prompt")
    if not template_string:
        raise ValueError(f"Prompt '{prompt_key}' in {prompts_path} is missing the 'prompt' key.")
    
    try:
        rendered_prompt = _env.from_string(template_string).render(**kwargs).strip()
    except Exception as e:
        raise ValueError(f"Error rendering prompt template: {e}")

    params = {k: v for k, v in prompt_config.items() if k != "prompt"}

    return rendered_prompt, params

def _parse_responses(
    resps,
    response_format: Optional[Type[BaseModel]] = None,
    strict_format: bool = True,
):
    results = []

    for resp in resps:
        try:
            content = resp.choices[0].message.get("content")
            if not response_format:
                results.append(content)
                continue
            results.append(response_format.model_validate_json(content).model_dump())
        except Exception as e:
            if strict_format:
                raise ValueError(
                    f"LLM returned invalid JSON.\n\nRaw content:\n{content}\n\nError: {e}"
                )
            else:
                results.append(None)
    return results

def run_llm(
    prompt: str | List[str],
    response_format: Type[BaseModel] | None = None,
    strict_format: bool = True,
    **kwargs: Any,
):
    prompts = [prompt] if isinstance(prompt, str) else prompt

    resps = batch_completion(
        messages=[[{"role": "user", "content": p}] for p in prompts],
        drop_params=True,
        response_format=response_format,
        **kwargs,
    )
    resps = _parse_responses(resps, response_format, strict_format)
    resps = resps[0] if isinstance(prompt, str) else resps
    return resps

async def run_llm_async(
    prompt: str | List[str],
    response_format: Type[BaseModel] | None = None,
    strict_format: bool = True,
    **kwargs: Any,
):
    prompts = [prompt] if isinstance(prompt, str) else prompt

    tasks = [
        acompletion(
            messages=[{"role": "user", "content": p}],
            response_format=response_format,
            **kwargs
        )
        for p in prompts
    ]
    
    resps = await asyncio.gather(*tasks)
    resps = _parse_responses(resps, response_format, strict_format)
    resps = resps[0] if isinstance(prompt, str) else resps
    return resps

def norm_special_characters(text):
    mapping = {'—': '-', '’': "'", '…': '...', '“': '"', '”': '"', '–': '-', '-': '-', '**': ''}
    pattern = re.compile("|".join(map(re.escape, mapping.keys())))
    return pattern.sub(lambda m: mapping[m.group(0)], text).strip()

def openai_batch(
    prompts,
    model="gpt-4.1-mini",
    reasoning_effort=None,
    max_tokens=None,
    api_key=None,
    completion_window="24h",
    metadata=None,
    poll_interval=60,
):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and not found in environment variables.")
    client = OpenAI(api_key=api_key)

    filename = f"batch_input_{uuid.uuid4().hex}.jsonl"
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            req = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            if max_tokens is not None:
                req["body"]["max_completion_tokens"] = max_tokens
            if reasoning_effort is not None:
                req["body"]["reasoning_effort"] = reasoning_effort
            f.write(json.dumps(req) + "\n")

    try:
        # ---- Step 3: Upload batch input file ----
        batch_input_file = client.files.create(
            file=open(filepath, "rb"),
            purpose="batch"
        )

        # ---- Step 4: Create the batch ----
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata or {"description": "Batch chat job"},
        )

        print(f"✅ Batch submitted: {batch.id}")
        print("⏳ Waiting for batch to complete...")

        completed_states = ["completed", "failed", "cancelled", "expired"]
        with tqdm(total=None) as pbar:
            while True:
                batch_status = client.batches.retrieve(batch.id)
                status = batch_status.status

                pbar.set_description(f"Batch status: {status}")
                pbar.update(1)

                if status in completed_states:
                    break
                
                time.sleep(poll_interval)
        
        if status != "completed":
            print(batch_status.errors.data[0])
            raise RuntimeError(f"Batch did not complete successfully (status: {status})")
        
        # ---- Step 6: Retrieve and parse results ----
        file_response = client.files.content(batch_status.output_file_id)
        raw_results = {}
        
        for line in file_response.text.strip().split("\n"):
            record = json.loads(line)
            cid = record.get("custom_id")
            content = None
            try:
                content = record["response"]["body"]["choices"][0]["message"]["content"]
                content = norm_special_characters(content)
            except Exception:
                print(f"⚠️ Warning: Failed to parse response for {cid}")
                pass
            raw_results[cid] = content

        # ---- Step 7: Return list of ordered results (content or None) ----
        results = []
        for i in range(len(prompts)):
            cid = f"request-{i+1}"
            results.append(raw_results.get(cid))

        print(f"✅ Batch completed: {len(results)} results retrieved.")
        return results

    finally:
        # ---- Step 8: Clean up temp file ----
        if os.path.exists(filepath):
            os.remove(filepath)

def gemini_batch(
    prompts,
    model="gemini-3-flash-preview",
    api_key=None,
    metadata=None,
    poll_interval=60,
    thinking_level='minimal',
):
    """
    Submit a batch job to Gemini API and wait for completion.
    
    Args:
        prompts: List of prompt strings
        model: Gemini model to use
        api_key: Gemini API key (or uses GEMINI_API_KEY env var)
        metadata: Optional metadata dict for the batch job
        poll_interval: Seconds to wait between status checks
        
    Returns:
        List of response strings (or None for failed requests), ordered by input
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not provided and not found in environment variables.")
    
    client = genai.Client(api_key=api_key)
    
    # ---- Step 1: Create JSONL file ----
    filename = f"gemini_batch_input_{uuid.uuid4().hex}.jsonl"
    filepath = os.path.join(os.getcwd(), filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            req = {
                "key": f"request-{i+1}",
                "request": {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }],
                    "generationConfig": {"thinkingConfig": {"thinkingLevel": thinking_level.upper()}}
                }
            }
            # if max_tokens is not None:
            #     req["request"]["generation_config"]["max_output_tokens"] = max_tokens
            # if thinking_budget is not None:
            #     req["request"]["generation_config"]["thinking_config"] = {"thinking_budget": thinking_budget}
            
            f.write(json.dumps(req) + "\n")
    
    try:
        # ---- Step 2: Upload file ----
        uploaded_file = client.files.upload(
            file=filepath,
            config=types.UploadFileConfig(
                display_name=f'batch-requests-{uuid.uuid4().hex[:8]}',
                mime_type='jsonl'
            )
        )
        print(f"📤 Uploaded file: {uploaded_file.name}")
        
        # ---- Step 3: Create batch job ----
        batch_config = {
            'display_name': metadata.get('description', 'Batch job') if metadata else 'Batch job'
        }
        
        batch_job = client.batches.create(
            model=model,
            src=uploaded_file.name,
            config=batch_config
        )
        
        print(f"✅ Batch submitted: {batch_job.name}")
        print("⏳ Waiting for batch to complete...")
        
        # ---- Step 4: Poll for completion ----
        completed_states = {
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED'
        }
        
        with tqdm(total=None) as pbar:
            while True:
                batch_status = client.batches.get(name=batch_job.name)
                status = batch_status.state.name

                print(f"Batch status: {status}")
                pbar.update(1)

                if status in completed_states:
                    break

                time.sleep(poll_interval)
        
        if status != 'JOB_STATE_SUCCEEDED':
            error_msg = f"Batch did not complete successfully (status: {status})"
            if batch_status.error:
                error_msg += f" - Error: {batch_status.error}"
            raise RuntimeError(error_msg)
        
        # ---- Step 5: Retrieve and parse results ----
        result_file_name = batch_status.dest.file_name
        print(f"📥 Downloading results from: {result_file_name}")
        
        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode('utf-8')
        
        raw_results = {}
        for line in file_content.splitlines():
            if not line.strip():
                continue
            
            record = json.loads(line)
            key = record.get("key")
            content = None
            
            try:
                if 'response' in record and record['response']:
                    # Extract text from response
                    candidates = record['response'].get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts and 'text' in parts[0]:
                            content = parts[0]['text']
                            content = norm_special_characters(content)
                elif 'error' in record:
                    print(f"⚠️ Warning: Error for {key}: {record['error']}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to parse response for {key}: {e}")
            
            raw_results[key] = content
        
        # ---- Step 6: Return ordered results ----
        results = []
        for i in range(len(prompts)):
            key = f"request-{i+1}"
            results.append(raw_results.get(key))
        
        print(f"✅ Batch completed: {len(results)} results retrieved.")
        return results
        
    finally:
        # ---- Step 7: Clean up temp file ----
        if os.path.exists(filepath):
            os.remove(filepath)

def get_embedding(
    text: str,
    model: str = "openai/text-embedding-3-large",
) -> List[float]:
    """
    Args:
        text (str): Input text to embed.
        model (str): Embedding model to use.
            - "text-embedding-3-small" (1536 dimensions)
            - "text-embedding-3-large" (3072 dimensions)
    Returns:
        list[float]: Embedding vector.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text input must be a non-empty string.")

    response = embedding(
        model=model,
        input=[text],
        encoding_format="float",
    )

    return response["data"][0]["embedding"]