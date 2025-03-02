import os
import re
import json
import random
from datetime import datetime, date, timedelta
from collections import defaultdict
from enum import Enum
from typing import Any, List, Mapping, Optional, Type

from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict

from langchain_aws import BedrockLLM, ChatBedrock

from cat.log import log
from cat.mad_hatter.decorators import tool, hook, plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3
from .bedrock_price_estimator import get_aws_service_pricing, parse_pricing_with_model


PLUGIN_NAME = "amazon_bedrock_llms"
DEFAULT_MODEL = "amazon.titan-tg1-large"

client = Boto3().get_client("bedrock")
pricing_client = Boto3().get_client("pricing")
bedrock_runtime_client = Boto3().get_client("bedrock-runtime")

PRICING_CACHE_FILE = os.path.join(
    MadHatter().plugins.get(PLUGIN_NAME)._path, "pricing_cache.json"
)

pricing_cache = {}


def load_pricing_cache():
    """Loads the pricing cache from a JSON file if available."""
    global pricing_cache
    if os.path.exists(PRICING_CACHE_FILE):
        try:
            with open(PRICING_CACHE_FILE, "r") as file:
                data = json.load(file)
                pricing_cache = {
                    model_id: {
                        "data": item["data"],
                        "timestamp": datetime.fromisoformat(item["timestamp"]),
                    }
                    for model_id, item in data.items()
                }
        except Exception as e:
            print(f"Failed to load pricing cache: {e}")
            pricing_cache = {}


def save_pricing_cache():
    """Saves the pricing cache to a JSON file."""
    try:
        with open(PRICING_CACHE_FILE, "w") as file:
            json.dump(
                {
                    model_id: {
                        "data": item["data"],
                        "timestamp": item["timestamp"].isoformat(),
                    }
                    for model_id, item in pricing_cache.items()
                },
                file,
                indent=4,
            )
    except Exception as e:
        print(f"Failed to save pricing cache: {e}")


load_pricing_cache()


def get_or_update_pricing(model_id, model_arn):
    if model_arn in pricing_cache:
        cached_data = pricing_cache[model_arn]
        if datetime.utcnow() - cached_data["timestamp"] < timedelta(days=1):
            return cached_data["data"]

    try:
        pricing_content = get_aws_service_pricing(
            "AmazonBedrock", pricing_client, model_id
        )
        model_pricing = parse_pricing_with_model(
            json.dumps(pricing_content), model_id, bedrock_runtime_client
        )

        pricing_cache[model_arn] = {
            "data": model_pricing,
            "timestamp": datetime.utcnow(),
        }
        return model_pricing

    except Exception as e:
        print(f"Error fetching pricing for {model_id}: {e}")
        return pricing_cache.get(model_id, {"error": "Pricing data unavailable"}).get(
            "data", {}
        )


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

        model_id = model["modelId"]
        pricing_info = get_or_update_pricing(model_id, model_arn)

        if "error" in pricing_info:
            continue

        if response_streaming_supported:
            models[modelName].append(
                {
                    "model_arn": model_arn,
                    "provider_name": provider_name,
                    "response_streaming_supported": response_streaming_supported,
                    "pricing_info": pricing_info,
                    "model_id": model_id,
                }
            )
            assert "." not in modelName

    save_pricing_cache()
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
    ClassBedrock = BedrockLLM if current_provider in ("cohere") else ChatBedrock

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
                _, _, guardrail_id, version = guardrail.split(":")
                guardrail_conf = {
                    "guardrailIdentifier": guardrail_id,
                    "guardrailVersion": version.replace("v", ""),
                    "trace": bool(kwargs.get("guardrail_trace", "False")),
                }
                input_kwargs["guardrails"] = guardrail_conf

            input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
            super(CustomBedrockLLM, self).__init__(**input_kwargs)

            if kwargs.get("budget_mode", "Disabled") != "Disabled":
                budget_limit = kwargs.get("budget_limit", "Unknown")
                input_price = kwargs.get("input_token_price", "Unknown")
                output_price = kwargs.get("output_token_price", "Unknown")
                setattr(
                    CustomBedrockLLM,
                    "_budget_config",
                    {
                        "budget_limit": (
                            float(budget_limit)
                            if isinstance(budget_limit, (int, float))
                            or budget_limit.replace(".", "", 1).isdigit()
                            else 0.0
                        ),
                        "input_token_price": (
                            float(input_price)
                            if isinstance(input_price, (int, float))
                            or input_price.replace(".", "", 1).isdigit()
                            else 0.0
                        ),
                        "output_token_price": (
                            float(output_price)
                            if isinstance(output_price, (int, float))
                            or output_price.replace(".", "", 1).isdigit()
                            else 0
                        ),
                        "budget_mode": kwargs.get("budget_mode", "Disabled"),
                    },
                )

        def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
            input_cost = (
                self._budget_config["input_token_price"] * input_tokens
            ) / 1000
            output_cost = (
                self._budget_config["output_token_price"] * output_tokens
            ) / 1000
            total_cost = input_cost + output_cost

            COST_CACHE_FILE = os.path.join(
                MadHatter().plugins.get(PLUGIN_NAME)._path, "COST_cache.json"
            )

            # Try to read existing cost cache, handle file errors safely
            pricing_cache = {}
            try:
                if os.path.exists(COST_CACHE_FILE):
                    with open(COST_CACHE_FILE, "r") as file:
                        pricing_cache = json.load(file) or {}
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load pricing cache. Error: {e}")
                pricing_cache = {}

            # Ensure "current_cost" is initialized properly
            pricing_cache["current_cost"] = float(
                pricing_cache.get("current_cost", 0.0)
            )

            # Update the cost
            pricing_cache["current_cost"] += total_cost

            # Try to persist the updated cost back to the file
            try:
                with open(COST_CACHE_FILE, "w") as file:
                    json.dump(pricing_cache, file, indent=4)
            except IOError as e:
                print(f"Error: Failed to save pricing cache. {e}")

            return round(pricing_cache["current_cost"], 6)

        def invoke(self, *args, **kwargs):
            response = super().invoke(*args, **kwargs)

            try:
                print(response.usage_metadata)
                print(self.compute_invocation_cost(**response.usage_metadata))
            except:
                pass

            return response

    CustomBedrockLLM.__name__ = class_name
    return CustomBedrockLLM


def get_model_price(llm_info):
    model_id = llm_info[0]["model_id"]
    pricing_info = llm_info[0].get("pricing_info", {})

    if isinstance(pricing_info, str):
        try:
            pricing_info = json.loads(pricing_info)
        except json.JSONDecodeError:
            print(
                f"Warning: Invalid pricing data format for {model_id}: {pricing_info}"
            )
            pricing_info = {}

    pricing_info = (
        pricing_info.get("rows", [pricing_info])[0]
        if isinstance(pricing_info, dict)
        else {}
    )

    input_token_price = pricing_info.get(model_id, {}).get("input", {})
    output_token_price = pricing_info.get(model_id, {}).get("output", {})

    return input_token_price, output_token_price


class BudgetMode(str, Enum):
    DISABLED = "Disabled"
    MONITOR = "Monitor"
    NOTIFY = "Notify"
    BLOCK = "Block"


def get_amazon_bedrock_llm_configs(amazon_llms, Guardrails, config_llms={}):
    for model_name, llm_info in amazon_llms.items():
        class_name = get_class_name(model_name)
        custom_bedrock_class = create_custom_bedrock_class(class_name, llm_info)

        input_token_price, output_token_price = get_model_price(llm_info)

        input_price = (
            input_token_price.get("price")
            if isinstance(input_token_price, dict)
            else None
        )
        input_price = (
            "Unknown" if input_price is None or input_price == 0.0 else input_price
        )

        output_price = (
            output_token_price.get("price")
            if isinstance(output_token_price, dict)
            else None
        )
        output_price = (
            "Unknown" if output_price is None or output_price == 0.0 else output_price
        )

        input_currency = input_token_price.get("currency")
        output_currency = output_token_price.get("currency")

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
            input_token_price: str = Field(
                default=input_price,
                description=f"The price per 1000 token (in {input_currency}).",
            )
            output_token_price: str = Field(
                default=output_price,
                description=f"The price per 1000 token (in {output_currency}).",
            )
            budget_mode: BudgetMode = Field(
                default=BudgetMode.DISABLED,
                description="The budget mode for the model. Options: Disabled, Monitor, Notify, Block.",
            )
            budget_limit: Optional[str] = Field(
                default="",
                description="The maximum budget for the model.",
            )
            _pyclass: Type = custom_bedrock_class
            model_config = ConfigDict(
                json_schema_extra={
                    "humanReadableName": f"Amazon Bedrock: {model_name}",
                    "description": f"Configuration for Amazon Bedrock LLMs ",
                    "link": "https://aws.amazon.com/bedrock/",
                },
                arbitrary_types_allowed=True,
                use_enum_values=True,
                validate_assignment=True,
                extra="allow",
                copy_on_model_validation="none",
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
            copy_on_model_validation="none",
        ),
    )
    return dynamic_model


_current_llms = []


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
            copy_on_model_validation="none",
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
