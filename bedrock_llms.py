from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3
from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from typing import Any, List, Mapping, Optional, Type
from cat.mad_hatter.mad_hatter import MadHatter
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock
from datetime import datetime, date
from collections import defaultdict
from enum import Enum
from cat.log import log
import random
import re
import json
from enum import Enum

PLUGIN_NAME = "amazon_bedrock_llms"
DEFAULT_MODEL = "amazon.titan-tg1-large"

client = Boto3().get_client("bedrock")


def get_availale_models(client):
    response = client.list_foundation_models(
        byOutputModality="TEXT", byInferenceType="ON_DEMAND"
    )
    models = defaultdict(list)
    for model in response["modelSummaries"]:
        model_arn = model["modelArn"]
        selected = model["providerName"].lower()
        provider_name = "mistral" if "mistral" in selected else selected
        response_streaming_supported = model["responseStreamingSupported"]
        model["modelName"] = model["modelName"].replace(".", ",")
        modelName = f"{model['providerName']} {model['modelName']}"
        if response_streaming_supported:
            models[modelName].append(
                {
                    "model_arn": model_arn,
                    "provider_name": provider_name,
                    "response_streaming_supported": response_streaming_supported,
                }
            )
            assert "." not in modelName
    return dict(models)


def get_available_guardrails(client):
    response = client.list_guardrails()
    guardrails = {"GUARDRAIL_0": "None"}
    index = 1
    for guardrail in response["guardrails"]:
        guardrail_details = client.list_guardrails(guardrailIdentifier=guardrail["id"])
        for detail in guardrail_details["guardrails"]:
            custom_identifier = (
                f'gr:{detail["name"]}:{detail["id"]}:v{detail["version"]}'
            )
            guardrails[f"GUARDRAIL_{index}"] = custom_identifier
            index += 1
    Guardrails = Enum("Guardrails", guardrails)
    return Guardrails


def get_class_name(name):
    class_name = re.sub("[^a-zA-Z0-9 \n\.]", "", name.lower()).title()
    class_name = class_name.replace(" ", "").replace(",", "o")
    assert "." not in class_name
    return f"CustomBedrockLLM{class_name}"


def create_custom_bedrock_class(class_name, llm_info):
    current_provider = llm_info[0]["provider_name"]
    ClassBedrock = BedrockLLM if current_provider == "cohere" else ChatBedrock

    class CustomBedrockLLM(ClassBedrock):
        def __init__(self, **kwargs):
            input_kwargs = {
                "model_id": llm_info[0]["model_arn"],
                "provider": llm_info[0]["provider_name"].lower(),
                "streaming": llm_info[0]["response_streaming_supported"],
                "model_kwargs": json.loads(kwargs.get("model_kwargs", "{}")),
                "client": Boto3().get_client("bedrock-runtime"),
            }
            guardrail = kwargs.get("guardrail_id", "None")
            if guardrail != "None":
                _, name, guardrail_id, version = guardrail.split(":")
                guardrail_conf = {
                    "guardrailIdentifier": guardrail_id,
                    "guardrailVersion": version.replace("v", ""),
                    "trace": bool(kwargs.get("guardrail_trace", "False")),
                }
                input_kwargs["guardrails"] = guardrail_conf
            input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
            super(CustomBedrockLLM, self).__init__(**input_kwargs)

    CustomBedrockLLM.__name__ = class_name
    return CustomBedrockLLM


def get_amazon_bedrock_llm_configs(amazon_llms, Guardrails, config_llms={}):
    for model_name, llm_info in amazon_llms.items():
        class_name = get_class_name(model_name)
        custom_bedrock_class = create_custom_bedrock_class(class_name, llm_info)

        class AmazonBedrockLLMConfig(LLMSettings):
            model_id: str = Field(
                default=llm_info[0]["model_arn"],
                description="The Amazon Resource Name (ARN) of the model.",
            )
            provider: str = Field(
                default=llm_info[0]["provider_name"],
                description="The name of the provider of the model.",
            )
            model_kwargs: Optional[str] = Field(
                default="{}",
                description="Additional keyword arguments for the model in JSON string format.",
            )
            guardrail_id: Guardrails = Field(
                default=Guardrails.GUARDRAIL_0,
                description="The guardrail setting to be applied to the model.",
            )
            guardrail_trace: Optional[bool] = Field(
                default=False,
                description="A boolean indicating whether to trace guardrail execution.",
            )
            _pyclass: Type = custom_bedrock_class
            model_config = ConfigDict(
                json_schema_extra={
                    "humanReadableName": f"Amazon Bedrock: {model_name}",
                    "description": "Configuration for Amazon Bedrock LLMs",
                    "link": "https://aws.amazon.com/bedrock/",
                },
                arbitrary_types_allowed=True,
                use_enum_values=True,
                validate_assignment=True,
                extra="allow",
                copy_on_model_validation="none"
            )

        new_class = type(class_name, (AmazonBedrockLLMConfig,), {})
        locals()[class_name] = new_class
        config_llms[model_name] = new_class
        assert "." not in class_name
        assert "." not in model_name
    return config_llms


def create_dynamic_model(amazon_llms) -> BaseModel:
    dynamic_fields = {}
    for model_name, llm_info in amazon_llms.items():
        model_name = model_name.replace(".", "o")
        dynamic_fields[model_name] = (
            bool,
            Field(
                default=llm_info[0]["model_arn"].endswith(DEFAULT_MODEL),
                description=f"Enable/disable the {model_name} model.",
            ),
        )
    dynamic_model = create_model(
        "DynamicModel", 
        **dynamic_fields,
        __config__=ConfigDict(
            arbitrary_types_allowed=True,
            use_enum_values=True,
            validate_assignment=True,
            extra="allow",
            copy_on_model_validation="none"
        )
    )
    return dynamic_model


_current_llms = []  # Global state for LLM settings

def get_settings():
    amazon_llms = get_availale_models(client)
    Guardrails = get_available_guardrails(client)
    config_llms = get_amazon_bedrock_llm_configs(amazon_llms, Guardrails)
    DynamicModel = create_dynamic_model(amazon_llms)

    class AmazonBedrockLLMSettings(DynamicModel):
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            use_enum_values=True,
            validate_assignment=True,
            extra="allow",
            copy_on_model_validation="none"
        )

        def init_llm(self):
            global _current_llms
            if not _current_llms:
                _current_llms = []

        def get_llms(self):
            global _current_llms
            return _current_llms

        @model_validator(mode="before")
        def validate(cls, values):
            global _current_llms
            _current_llms = []
            for llm in values.keys():
                if llm in values and values[llm]:
                    _current_llms.append(config_llms[llm])
            log.info("Dynamically Selected LLMs:")
            log.info(
                [
                    llm.model_config["json_schema_extra"]["humanReadableName"]
                    for llm in _current_llms
                ]
            )
            return values

    return AmazonBedrockLLMSettings


@plugin
def settings_model():
    return get_settings()


def factory_pipeline():
    AmazonBedrockLLMSettings = get_settings()
    settings = AmazonBedrockLLMSettings()
    settings.init_llm()
    aws_plugin = MadHatter().plugins.get(PLUGIN_NAME)
    plugin_settings = aws_plugin.load_settings()
    AmazonBedrockLLMSettings(**plugin_settings)
    return settings.get_llms()


@hook
def factory_allowed_llms(allowed, cat) -> List:
    return allowed + factory_pipeline()
