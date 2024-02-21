import os
import openai
import requests
from tenacity import (
    retry, 
    retry_if_exception_type, 
    retry_if_not_exception_message, 
    wait_random_exponential, 
    stop_after_attempt
)


class API:
    retry_kargs = dict(
        retry=retry_if_exception_type((
            IOError,
            openai.OpenAIError,
        )) & retry_if_not_exception_message(
            match='You exceeded your current quota, please check your plan and billing details.'
        ) & retry_if_not_exception_message(
            match='Your account is not active, please check your billing details on our website.'
        ) & retry_if_not_exception_message(
            match='no model permission: (.*)'
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(100)
    )

    default_api_key = os.getenv('OPENAI_API_KEY')
    default_api_type = os.getenv('OPENAI_API_TYPE')
    default_api_version = os.getenv('OPENAI_API_VERSION')
    default_api_base = os.getenv('OPENAI_API_BASE')

    models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k']
    price_rate = {
        'gpt-3.5-turbo': {
            'prompt_tokens': 0.002, 
            'completion_tokens': 0.0015
        },
        'gpt-3.5-turbo-0613': {
            'prompt_tokens': 0.002, 
            'completion_tokens': 0.0015
        },
        'gpt-3.5-turbo-0301': {
            'prompt_tokens': 0.002, 
            'completion_tokens': 0.0015
        },
        'gpt-3.5-turbo-instruct': {
            'prompt_tokens': 0.002, 
            'completion_tokens': 0.0015
        },
        'gpt-3.5-turbo-16k': {
            'prompt_tokens': 0.004, 
            'completion_tokens': 0.003
        },
        'gpt-3.5-turbo-16k-0613': {
            'prompt_tokens': 0.004, 
            'completion_tokens': 0.003
        },
        'gpt-4': {
            'prompt_tokens': 0.06, 
            'completion_tokens': 0.03
        },
        'gpt-4-0613': {
            'prompt_tokens': 0.06, 
            'completion_tokens': 0.03
        },
        'gpt-4-32k': {
            'prompt_tokens': 0.12, 
            'completion_tokens': 0.06
        },
        'text-embedding-ada-002': {
            'prompt_tokens': 0.0001,
        }
    }
    
    def __init__(self, api_key=None, api_type=None, api_base=None, api_version=None):
        self.api_key = api_key or self.default_api_key
        self.api_type = api_type or self.default_api_type
        self.api_base = api_base or self.default_api_base
        self.api_version = api_version or self.default_api_version
        self.available_models = self.models # NOT IMPLEMENTED
        self.usage = {
            model: {
                'prompt_tokens': 0, 
                'completion_tokens': 0
            } 
            for model in self.available_models
        }
        self.setup()
    
    def setup(self):
        openai.api_key = self.api_key
        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
    
    def validate(self):
        if self.api_type == 'azure':
            self.available_models = self.models # NOT IMPLEMENTED
        else:
            for model in self.models:
                validation_url = f'https://api.openai.com/v1/models/{model}'
                headers = {"Authorization": f"Bearer {self.api_key}"}
                r = requests.get(validation_url, headers=headers)
                if r.status_code == 401:
                    raise ValueError(f"Invalid API, the current key: {self.api_key}")
                elif r.status_code != 404:
                    self.available_models.append(model)

    @retry(**retry_kargs)
    def completion(self, prompt, **kargs):
        assert isinstance(prompt, str), 'Completion does NOT support dialogue'
        responses = openai.Completion.create(prompt=prompt, **kargs)
        texts = [response.text for response in responses.choices]
        return texts, responses.usage
    
    @retry(**retry_kargs)
    def chat_completion(self, messages, **kargs):
        responses = openai.ChatCompletion.create(messages=messages, **kargs)
        contents = [response.message.content for response in responses.choices]
        return contents, responses.usage

    @retry(**retry_kargs)
    def embedding(self, text: str, model: str = 'text-embedding-ada-002') -> list[float]:
        cfg = dict(input=text, engine=model) if self.api_type == 'azure' else dict(input=text, model=model)            
        return openai.Embedding.create(**cfg).data[0].embedding

    def gpt(self, query, model='gpt-4', **kargs):
        self.setup()
        if self.api_type == 'azure':
            kargs['engine'] = model.replace('.', '')
        else:
            kargs['model'] = model

        if 'turbo-instruct' in model:
            responses, usage = self.completion(query, **kargs)
        else:
            if isinstance(query, str):
                messages = [dict(role='user', content=query)]
            elif isinstance(query, list):
                messages = query
            else:
                raise TypeError(f'Your query should be either dialogues (list) or a prompt (str): {query}')
            responses, usage = self.chat_completion(messages, **kargs)
        
        # log completion tokens
        # self.usage[model]['prompt_tokens'] += usage.prompt_tokens
        # self.usage[model]['completion_tokens'] += usage.completion_tokens
        
        return responses
        
    def gpt_usage(self):
        usage = {}
        for model in self.available_models:
            if sum(self.usage[model].values()) > 0:
                usage[model] = 1e-3 * sum(
                    self.usage[model][key] * self.price_rate[model][key] 
                    for key in self.usage[model]
                )
        return usage
