import os
import re
from typing import List, Tuple, Optional

from llm.code import check_valid_program
from llm.prompt import env_prompt, sys_prompt, generate_prompt, hyp_prompt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM:
    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        self.h_idx = 0
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif 'deepseek' in model:
            self.client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
        elif 'qwen' in model:
            self.client = OpenAI(
                api_key=os.getenv('QWEN_API_KEY'),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.messages = [{"role": "system", "content": sys_prompt}]
        self.chat_initialized = False

        # counters for logging
        self.last_generate_invalid_count = 0
        self.last_refine_invalid_count = 0

        # hard caps
        self.MAX_MESSAGES = 500
        self.MAX_REFINE_RETRIES = 5
        self.MAX_GENERATE_RETRIES = 5

    def reset_chat(self):
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.chat_initialized = False
        self.last_generate_invalid_count = 0
        self.last_refine_invalid_count = 0

    def _clean_response(self, text: str) -> str:
        code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return text.strip()

    def _chat_completion(self) -> str:
        print("\n[LLM CALL START]")
        print(f"Model: {self.model}")
        print(f"Messages length: {len(self.messages)}")

        if len(self.messages) >= self.MAX_MESSAGES:
            raise RuntimeError(
                f"MESSAGE_LIMIT_ABORT: messages length {len(self.messages)} exceeded limit {self.MAX_MESSAGES}"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            print("[LLM ERROR]", e)
            raise

        raw_text = response.choices[0].message.content

        print("\n[LLM RAW OUTPUT]")
        print(raw_text)

        text = self._clean_response(raw_text)

        print("\n[LLM CLEANED OUTPUT]")
        print(text)

        self.messages.append({"role": "assistant", "content": text})

        print("[LLM CALL END]\n")
        return text

    def generate(self, evidence: List) -> Tuple[str, str]:
        if not self.chat_initialized:
            user_prompt = env_prompt + hyp_prompt + generate_prompt
            self.messages.append({"role": "user", "content": user_prompt})
            self.chat_initialized = True

        invalid_count = 0

        while invalid_count < self.MAX_GENERATE_RETRIES:
            print(f"\n[GENERATE ATTEMPT {invalid_count + 1}]")
            hypothesis = self._chat_completion()

            print("\n[VALIDATING PROGRAM]")
            if check_valid_program(hypothesis):
                print("[VALID PROGRAM ACCEPTED]")
                self.h_idx += 1
                self.last_generate_invalid_count = invalid_count
                return hypothesis, f'generated_{self.h_idx}'

            print("[INVALID PROGRAM REJECTED]")
            invalid_count += 1

            self.messages.append({
                "role": "user",
                "content": (
                    "Your previous output was not valid executable Python. "
                    "Output only a complete valid Python program containing exactly:\n\n"
                    "def predict(key, box):\n"
                    "    return ...\n\n"
                    "No comments. No explanation. No markdown."
                )
            })

        self.last_generate_invalid_count = invalid_count
        print(f"[GENERATE ABORT] exceeded {self.MAX_GENERATE_RETRIES} invalid generate attempts")
        raise RuntimeError(
            f"GENERATE_ABORT: exceeded {self.MAX_GENERATE_RETRIES} invalid generate attempts"
        )

    def refine(self, evidence: List, old_h: str) -> Tuple[Optional[str], Optional[str]]:
        assert len(evidence) > 0, "Refine should only be called after at least one trial."

        key, box, outcome = evidence[-1]

        opened_boxes = sorted({b.id for (_, b, o) in evidence if o})
        opened_boxes_text = ", ".join(opened_boxes) if opened_boxes else "none"

        outcome_text = (
            f"Trial result: Key {key.id} successfully opened box {box.id}."
            if outcome is True
            else f"Trial result: Key {key.id} failed to open box {box.id}."
        )

        user_prompt = f"""
Your previous hypothesis was:

{old_h}

{outcome_text}

Boxes opened so far: {opened_boxes_text}
Number of boxes opened so far: {len(opened_boxes)}

Please refine your hypothesis based on the latest evidence.
Remember that the correct key may still fail on some trials due to mechanical failure.
Output only a complete valid Python program containing exactly:

def predict(key, box):
    return ...

No comments. No explanation. No markdown.
""".strip()

        self.messages.append({"role": "user", "content": user_prompt})

        invalid_count = 0

        while invalid_count < self.MAX_REFINE_RETRIES:
            print(f"\n[REFINE ATTEMPT {invalid_count + 1}]")

            hypothesis = self._chat_completion()

            print("\n[VALIDATING PROGRAM]")
            if check_valid_program(hypothesis):
                print("[VALID PROGRAM ACCEPTED]")
                self.h_idx += 1
                self.last_refine_invalid_count = invalid_count
                return hypothesis, f'generated_{self.h_idx}'

            print("[INVALID PROGRAM REJECTED]")
            invalid_count += 1

            self.messages.append({
                "role": "user",
                "content": (
                    "Your previous output was not valid executable Python. "
                    "Output only a complete valid Python program containing exactly:\n\n"
                    "def predict(key, box):\n"
                    "    return ...\n\n"
                    "No comments. No explanation. No markdown."
                )
            })

        self.last_refine_invalid_count = invalid_count
        print(f"[REFINE ABORT] exceeded {self.MAX_REFINE_RETRIES} invalid refine attempts")
        raise RuntimeError(
            f"REFINE_ABORT: exceeded {self.MAX_REFINE_RETRIES} invalid refine attempts"
        )