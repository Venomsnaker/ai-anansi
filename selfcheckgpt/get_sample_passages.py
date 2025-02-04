from openai import OpenAI
from tqdm import tqdm

class GetSamplePassages:
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
        self.prompt_template="Please paraphrase the following text, correct any factual inaccuracies and ensuring all the information remains intact: {passage}\nOnly return the corrected version with the same number of sentences."
        
    def set_prompt_template(self, prompt_template: str):
        self.prompt_template=prompt_template
        
    def get_respond(self, prompt: str):
        if self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1.0,
            )
            return response.choices[0].message.content
        else:
            raise ValueError("client_type not implemented")
        
    def get_sample_passages(
        self,
        input_text: str,
        sample_passages_size = 5, 
        verbose: bool = False,
    ):
        sample_passages = []
        input_text = input_text.strip()
        prompt = self.prompt_template.format(passage=input_text)
        disable = not verbose
        
        for i in tqdm(range(sample_passages_size), disable=disable):
            generated_text = self.get_respond(prompt=prompt)
            sample_passages.append(generated_text)
        return sample_passages