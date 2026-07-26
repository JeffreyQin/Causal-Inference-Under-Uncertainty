import os
import re
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI
from llm.prompt import env_prompt, sys_prompt, generate_prompt, refine_prompt, hyp_prompt

load_dotenv()


class LLM:
    def __init__(
        self,
        model: str = "deepseek-chat",  # or "qwen3.6-plus"
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        self.h_idx = 0
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if "gpt" in model:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "deepseek" in model:
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        elif "qwen" in model:
            self.client = OpenAI(
                api_key=os.getenv("QWEN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _clean_response(self, text: Optional[str]) -> str:
        text = text or ""
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

    def get_completion(self, sys_prompt_text: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt_text},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._clean_response(response.choices[0].message.content)

    def generate_once(self, evidence: List) -> Tuple[str, str]:
        """Exactly one API call. No internal retry loop."""
        if len(evidence) != 0:
            raise ValueError("generate_once currently expects empty evidence.")
        user_prompt = env_prompt + hyp_prompt + generate_prompt
        hypothesis = self.get_completion(sys_prompt, user_prompt)
        self.h_idx += 1
        return hypothesis, f"generated_{self.h_idx}"

    def refine_once(self, evidence: List, old_h: str) -> Tuple[str, str]:
        """Exactly one API call. No internal retry loop."""
        user_prompt = env_prompt + hyp_prompt + refine_prompt[0] + old_h + refine_prompt[1]
        for (key, box, outcome) in evidence:
            if outcome is True:
                user_prompt += f"Key {key.id} successfully opened box {box.id}\n"
            else:
                user_prompt += f"Key {key.id} failed to open box {box.id}\n"
        user_prompt += refine_prompt[2]

        hypothesis = self.get_completion(sys_prompt, user_prompt)
        self.h_idx += 1
        return hypothesis, f"generated_{self.h_idx}"
