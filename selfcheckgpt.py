from openai import OpenAI
from tqdm import tqdm
import numpy as np
from typing import List
import os

class SelfCheckGPT:
    def __init__(
        self,
        client_type="openai",
        model="solar-pro",
        base_url="https://api.upstage.ai/v1/solar",
        api_key=None,
    ):
        if client_type == "openai":
            self.client=OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        self.client_type=client_type
        self.model=model
        self.prompt_template="Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping={'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text=set()
        
    def set_prompt_template(self, prompt_template: str):
        self.prompt_template=prompt_template
        
    def completion(self, prompt: str):
        if self.client_type == "openai":
            chat_result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=5
            )
            return chat_result['choices'][0]['message']['content']
        else:
            raise ValueError("client_type not implemented")
        
    def chat_result_postprocessing(self, text: str):
        text = text.lower().strip()
        
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"Warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
    
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        n_sentences = len(sentences)
        n_samples = len(sampled_passages)
        scores = np.zeros((n_sentences, n_samples))
        disable = not verbose
        
        for sent_i in tqdm(range(n_sentences), disable = disable):
            sentence = sentences[sent_i]
            
            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_text = self.completion(prompt=prompt)
                score = self.chat_result_postprocessing(text=generate_text)
                scores[sent_i, sample_i] = score
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence
            