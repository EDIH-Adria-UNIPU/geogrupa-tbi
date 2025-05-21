import base64
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
Your task is to classify traffic signs based on the provided image, in JSON format.
The options for classification are:
- Stop sign
- Priority road sign
- Other sign
- Not a sign
"""


class SignClassification(BaseModel):
    category: Literal["stop_sign", "priority_road_sign", "other_sign", "not_a_sign"]


def classify_sign(image_path: Path) -> str:
    image_data = image_path.read_bytes()
    base64_image = base64.b64encode(image_data).decode("utf-8")
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                ],
            },
        ],
        text_format=SignClassification,
    )
    return response.output_parsed.category
