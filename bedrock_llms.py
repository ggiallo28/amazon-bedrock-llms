from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3
from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from typing import Any, List, Mapping, Optional, Type
from cat.mad_hatter.mad_hatter import MadHatter
from langchain_aws import BedrockLLM
from datetime import datetime, date
from collections import defaultdict
from enum import Enum
from cat.log import log
import random
import re
import json
from enum import Enum

    
DEFAULT_MODEL =  "amazon.titan-tg1-large"

client = Boto3().get_client("bedrock")

def get_availale_models(client):
    response = client.list_foundation_models(
        byOutputModality="TEXT",
        byInferenceType='ON_DEMAND'
    )
    models = defaultdict(list)
    for model in response["modelSummaries"]:
        model_arn = model['modelArn']
        selected = model['providerName'].lower()
        provider_name = 'mistral' if 'mistral' in selected else selected
        response_streaming_supported = model['responseStreamingSupported']
        modelName = f"{model['providerName']} {model['modelName']}"
        if response_streaming_supported:
            models[modelName].append({
                "model_arn": model_arn,
                "provider_name": provider_name,
                "response_streaming_supported": response_streaming_supported
            })
    return dict(models)

def get_class_name(name):
    class_name = re.sub('[^a-zA-Z0-9 \n\.]', '', name.lower()).title()
    class_name = class_name.replace(" ", "")
    return f"CustomBedrockLLM{class_name}"

def create_custom_bedrock_class(class_name, llm_info):
    class CustomBedrockLLM(BedrockLLM):
        def __init__(self, **kwargs):
            input_kwargs = {
                "model_id": llm_info[0]["model_arn"],
                "provider": llm_info[0]["provider_name"].lower(),
                "streaming": llm_info[0]["response_streaming_supported"],
                "model_kwargs": json.loads(kwargs.get("model_kwargs", "{}")),
                "client": Boto3().get_client("bedrock-runtime")
            }
            input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
            super(CustomBedrockLLM, self).__init__(**input_kwargs)
            
    CustomBedrockLLM.__name__ = class_name
    return CustomBedrockLLM

def get_amazon_bedrock_llm_configs(amazon_llms, config_llms={}):
    for model_name, llm_info in amazon_llms.items():
        class_name = get_class_name(model_name)
        custom_bedrock_class = create_custom_bedrock_class(class_name, llm_info)
        class AmazonBedrockLLMConfig(LLMSettings):
            model_id: str = llm_info[0]["model_arn"]
            provider: str = llm_info[0]["provider_name"]
            model_kwargs: str = "{}"
            _pyclass: Type = custom_bedrock_class
            
            model_config = ConfigDict(
                json_schema_extra={
                    "humanReadableName":  f"Amazon Bedrock: {model_name}",
                    "description": "Configuration for Amazon Bedrock LLMs",
                    "link": "https://aws.amazon.com/bedrock/",
                }
            )
        
        new_class = type(class_name, (AmazonBedrockLLMConfig,), {})
        locals()[class_name] = new_class
        config_llms[model_name] = new_class
    return config_llms
    
def create_dynamic_model(amazon_llms) -> BaseModel:
    dynamic_fields = {}
    for model_name, llm_info in amazon_llms.items():
        dynamic_fields[model_name] = (
            bool,
            Field(
                default=llm_info[0]["model_arn"].endswith(DEFAULT_MODEL), 
                description=f"Enable/disable the {model_name} model."
            ),
        )
    dynamic_model = create_model("DynamicModel", **dynamic_fields)
    return dynamic_model

amazon_llms = get_availale_models(client)
config_llms = get_amazon_bedrock_llm_configs(amazon_llms)
DynamicModel = create_dynamic_model(amazon_llms)
class AmazonBedrockLLMSettings(DynamicModel):
    @classmethod
    def init_llm(cls):
        if not hasattr(cls, '_current_llms'):
            setattr(cls, '_current_llms', [])
    @classmethod
    def get_llms(cls):
        return cls._current_llms
    @model_validator(mode="before")
    def validate(cls, values):
        cls._current_llms = []
        for emb in values.keys():
            if values[emb]:
                cls._current_llms.append(config_llms[emb])
        log.info("Dynamically Selected LLMs:")
        log.info([
            llm.model_config['json_schema_extra']['humanReadableName']
            for llm in cls._current_llms
        ])
        return values

@plugin
def settings_model():
    return AmazonBedrockLLMSettings
    
@hook
def factory_allowed_llms(allowed, cat) -> List:
    AmazonBedrockLLMSettings.init_llm()
    aws_plugin = MadHatter().plugins.get("bedrock_models")
    plugin_settings = aws_plugin.load_settings()
    AmazonBedrockLLMSettings(**plugin_settings)
    return allowed + AmazonBedrockLLMSettings.get_llms()