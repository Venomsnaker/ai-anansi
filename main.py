import selfcheckgpt
import torch
import os
from dotenv import load_dotenv
import warnings
import re

import selfcheckgpt.get_sample_passages
import selfcheckgpt.selfcheckgpt
warnings.filterwarnings("ignore")

SENTENCE_ENDINGS = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'


def check_hallucination_in_text(selfcheckgpt_prompt: selfcheckgpt.selfcheckgpt.SelfCheckGPT, sampled_passages_getter: selfcheckgpt.get_sample_passages.GetSamplePassages, text: str):
    text = text.strip()
    sentences = re.split(SENTENCE_ENDINGS, text)
    print(sentences)
    sampled_text = sampled_passages_getter.get_sample_passages(input_text=text)
    print("Finish getting sampled_text:\n", sampled_text)
    
    scores = selfcheckgpt_prompt.predict(
        sentences=sentences,
        sampled_passages=sampled_text,
        verbose=True
    )
    print("Finish getting scores:\n", scores)
    # Predict using scores


if __name__ == "__main__":
    # Setup
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheckgpt_prompt = selfcheckgpt.selfcheckgpt.SelfCheckGPT(
        client_type="openai",
        model="solar-pro",
        base_url="https://api.upstage.ai/v1/solar",
        api_key=api_key
    )

    sampled_passages_getter = selfcheckgpt.get_sample_passages.GetSamplePassages(
        client_type="openai",
        model="solar-pro",
        base_url="https://api.upstage.ai/v1/solar",
        api_key=api_key
    )

    check_hallucination_in_text(
        selfcheckgpt_prompt=selfcheckgpt_prompt,
        sampled_passages_getter=sampled_passages_getter,
        text="""
            Seiko 5 is the most valuable Seiko watches of all times. Meanwhiles, Grand Seiko is not that great.
        """
    )
