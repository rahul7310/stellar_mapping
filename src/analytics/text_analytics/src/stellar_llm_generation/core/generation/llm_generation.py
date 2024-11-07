"""
A module for generating responses from a language model.
"""

from typing import List
import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("stellar_llm_generation/.env")

class LLMGenerator:
    """
    A class for generating responses from a language model.
    """
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.api_key = os.getenv("MODEL_API_KEY")

    def generate(self, messages: List[str]) -> str:
        """
        Generate a response from the model given a list of messages.

        Args:
            messages (List[str]): A list of messages to send to the model.

        Returns:
            str: The response from the model.
        """
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        completion = client.chat.completions.create(
            #             extra_headers={
                                        #         "HTTP-Referer": $YOUR_SITE_URL,  # Optional, for including your app on openrouter.ai rankings.
                                        # "X-Title": $YOUR_APP_NAME,  # Optional. Shows in rankings on openrouter.ai.
            # },
            model = self.model_name,
            messages = [{"role": "user", "content": message} for message in messages]
        )
        return completion.choices[0].message.content
