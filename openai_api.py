from openai import AsyncOpenAI
from typing import Dict, Optional
from transformers import AutoTokenizer

__ai = AsyncOpenAI(api_key="hello", base_url="http://localhost:8081/v1")


async def openai_api(prompt: str, extra_args: Optional[Dict[str, str]] = None):
    res = await __ai.completions.create(
        prompt=prompt,
        model="default",
        max_tokens=2048,
        stream=True,
        extra_body=extra_args,
    )
    content = ""
    async for x in res:
        x = x.content if hasattr(x, "content") else x.choices[0].text
        content += x
        yield {"part": x}
    yield {"all": content.strip()}


tok = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
tok.chat_template = """
{% for message in messages %}
    {% if (message['role'] == 'assistant') %}
        {% set role = 'model' %}{% else %}{% set role = message['role'] %}
    {% endif %}
    {{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{'<start_of_turn>model\n'}}
{% endif %}
"""
