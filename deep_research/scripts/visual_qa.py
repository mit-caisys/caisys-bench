"""
Visual QA tool for answering questions about images using multimodal LLMs.
"""

import base64
import mimetypes
import os
import uuid
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from smolagents import tool

load_dotenv(override=True)


def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_visualizer_tool(llm):
    """
    Factory function to create a configured visualizer tool.
    """

    @tool
    def visualizer(image_path: str, question: str | None = None) -> str:
        """A tool that can answer questions about attached images using Gemma 3.

        Args:
            image_path: The path to the image on which to answer the question.
            question: The question to answer.
        """

        add_note = False
        if not question:
            add_note = True
            question = "Please describe this image in detail."

        mime_type, _ = mimetypes.guess_type(image_path)
        base64_image = encode_image(image_path)

        payload = {
            "model": llm.model.split("/", 1)[-1],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": llm.max_tokens,
            "temperature": llm.temperature,
        }

        vllm_url = f"{llm.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(vllm_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            output = result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Failed to connect to vLLM at {vllm_url}. Error: {str(e)}")

        if add_note:
            output = f"Detailed caption: {output}"

        return output

    return visualizer


# For Testing
@dataclass
class LLM:
    model: str
    max_tokens: int
    temperature: float
    api_base: str


if __name__ == "__main__":
    llm = LLM(
        "hosted_vllm/google/gemma-3-27b-it", 1000, 0.2, "http://localhost:8000/v1"
    )
    visual = create_visualizer_tool(llm)
    print(visual("downloads/a.jpg", "How many people are there in this picture?"))
