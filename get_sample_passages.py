from openai import OpenAI

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
        self.prompt_template="""
            Please paraphrase the following text, correct any factual inaccuracies and ensuring all the information remains intact: {passage}.
            Only return the corrected version with the same number of sentences.
        """
        
    def set_prompt_template(self, prompt_template: str):
        self.prompt_template=prompt_template
        
    def get_respond(self, prompt: str):
        if self.