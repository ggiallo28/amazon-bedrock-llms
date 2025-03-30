import os
import re
import json
import random
from datetime import datetime, date, timedelta
from collections import defaultdict
from enum import Enum
from typing import Any, List, Mapping, Optional, Type
from cat.mad_hatter.decorators import tool
from cat.db import crud

from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from langchain_core.messages import AIMessage
from cat.mad_hatter.decorators import hook

from langchain_aws import BedrockLLM, ChatBedrock
import logging

from cat.log import log
from cat.mad_hatter.decorators import tool, hook, plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3
from .bedrock_price_estimator import (
    fetch_aws_pricing,
    parse_pricing_with_model,
    get_model_names,
    filter_pricing_by_model,
    extract_model_pricing,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PLUGIN_NAME = "amazon_bedrock_llms"
DEFAULT_MODEL = "amazon.titan-tg1-large"

CACHED_PRICING_FILE = os.path.join(
    MadHatter().plugins.get(PLUGIN_NAME)._path, "cached_model_pricing.json"
)
CACHED_COST_FILE = os.path.join(
    MadHatter().plugins.get(PLUGIN_NAME)._path, "cached_model_costs.json"
)

client = Boto3().get_client("bedrock")
pricing_client = Boto3().get_client("pricing")
bedrock_runtime_client = Boto3().get_client("bedrock-runtime")

pricing_cache = {}


def load_pricing_cache():
    """Loads the pricing cache from a JSON file if available."""
    global pricing_cache
    if os.path.exists(CACHED_PRICING_FILE):
        try:
            with open(CACHED_PRICING_FILE, "r") as file:
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
        with open(CACHED_PRICING_FILE, "w") as file:
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

    model_pricing = {model_id: {"input": {}, "output": {}, "cache_read_input": {}}}

    try:
        pricing_data = fetch_aws_pricing(pricing_client)

        model_names = get_model_names(pricing_data)

        model_name = parse_pricing_with_model(
            model_names, model_id, bedrock_runtime_client
        )

        if model_name == "Error":
            pricing_cache[model_arn] = {
                "data": model_pricing,
                "timestamp": datetime.utcnow(),
            }
            return {"error": "Failed to determine model name."}

        filtered_pricing = filter_pricing_by_model(pricing_data, model_name)

        if filtered_pricing:
            model_pricing = extract_model_pricing(filtered_pricing, model_id)

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
                input_token_unit = kwargs.get("input_token_unit", "Unknown")
                output_token_unit = kwargs.get("output_token_unit", "Unknown")

                def parse_float(value, default=0.0):
                    if isinstance(value, (int, float)):
                        return float(value)
                    return (
                        float(value) if value.replace(".", "", 1).isdigit() else default
                    )

                budget_limit = parse_float(budget_limit)
                input_price = parse_float(input_price)
                output_price = parse_float(output_price)
                input_token_unit = parse_float(input_token_unit, default=1.0)
                output_token_unit = parse_float(output_token_unit, default=1.0)

                setattr(
                    CustomBedrockLLM,
                    "_budget_config",
                    {
                        "budget_limit": budget_limit,
                        "input_token_price": input_price / input_token_unit,
                        "output_token_price": output_price / output_token_unit,
                        "budget_mode": kwargs.get("budget_mode", "Disabled"),
                    },
                )

        def get_current_model_cost(self):
            """Retrieves the total model cost from the cache file."""
            if os.path.exists(CACHED_COST_FILE):
                try:
                    with open(CACHED_COST_FILE, "r") as file:
                        pricing_cache = json.load(file) or {}
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load pricing cache. Error: {e}")
                    pricing_cache = {}
            else:
                pricing_cache = {}

            return float(pricing_cache.get("current_cost", 0.0))

        def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
            """Computes cost for the current request and updates the total model cost."""
            budget_config = getattr(self, "_budget_config", {})
            input_price = budget_config.get("input_token_price", 0.0)
            output_price = budget_config.get("output_token_price", 0.0)

            input_cost = input_price * input_tokens
            output_cost = output_price * output_tokens
            current_request_cost = round(input_cost + output_cost, 6)

            model_total_cost = self.get_current_model_cost() + current_request_cost

            pricing_cache = {"current_cost": model_total_cost}
            try:
                with open(CACHED_COST_FILE, "w") as file:
                    json.dump(pricing_cache, file, indent=4)
            except IOError as e:
                logger.error(f"Error saving pricing cache: {e}")

            return model_total_cost, current_request_cost

        def invoke(self, *args, **kwargs):
            budget_config = getattr(self, "_budget_config", {})
            budget_mode = str(
                budget_config.get("budget_mode", BudgetMode.DISABLED)
            ).capitalize()
            budget_limit = float(budget_config.get("budget_limit", 0.0))

            model_total_cost = self.get_current_model_cost()

            alert_message = ""
            if budget_limit > 0 and model_total_cost > budget_limit:
                alert_message = (
                    f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
                )
                logger.warning(alert_message)

            if (
                budget_mode == BudgetMode.BLOCK.value
                and model_total_cost > budget_limit
            ):
                return AIMessage(
                    content="⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                    "Your request cannot be processed because the total cost has exceeded the budget limit.\n"
                    "💰 **Cost Breakdown:**\n"
                    f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                    f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
                )

            response = super().invoke(*args, **kwargs)

            try:
                usage_metadata = response.usage_metadata
                model_total_cost, current_request_cost = self.compute_invocation_cost(
                    **usage_metadata
                )

                response.usage_metadata["current_request_cost"] = round(
                    current_request_cost, 6
                )
                response.usage_metadata["model_total_cost"] = round(model_total_cost, 6)

                if budget_mode == BudgetMode.MONITOR.value:
                    logger.info(f"Invocation Cost: ${current_request_cost:.6f}")
                    logger.info(f"Total Cost (All Calls): ${model_total_cost:.6f}")
                    if alert_message:
                        logger.warning(alert_message)

                if budget_mode == BudgetMode.NOTIFY.value and alert_message:
                    response.content += f"\n\n🚨 **{alert_message}** 🚨\n"

                if budget_mode == BudgetMode.TRACE.value:
                    response.content += "\n\n"
                    if alert_message:
                        response.content += f"🚨 **{alert_message}** 🚨\n"
                    response.content += (
                        f"💰 **Cost Breakdown:**\n"
                        f"   - 📝 Request Cost: `${current_request_cost:.6f}`\n"
                        f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
                    )

                response.usage_metadata["budget_mode"] = budget_mode

            except Exception as e:
                logger.error(f"Error processing cost computation: {e}")

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
    TRACE = "Trace"
    BLOCK = "Block"


def get_amazon_bedrock_llm_configs(amazon_llms, Guardrails, config_llms={}):
    for model_name, llm_info in amazon_llms.items():
        class_name = get_class_name(model_name)
        custom_bedrock_class = create_custom_bedrock_class(class_name, llm_info)

        input_token_price, output_token_price = get_model_price(llm_info)

        input_price = (
            input_token_price.get("price", "0.0")
            if isinstance(input_token_price, dict)
            else None
        )
        input_price = (
            "Unknown" if input_price is None or input_price == 0.0 else input_price
        )

        output_price = (
            output_token_price.get("price", "0.0")
            if isinstance(output_token_price, dict)
            else None
        )
        output_price = (
            "Unknown" if output_price is None or output_price == 0.0 else output_price
        )

        input_currency = input_token_price.get("currency", "USD")
        output_currency = output_token_price.get("currency", "USD")
        input_unit = input_token_price.get("unit", "0.0")
        output_unit = output_token_price.get("unit", "0.0")

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
                description=f"The price per {input_unit} token (in {input_currency}).",
            )
            input_token_unit: str = Field(
                default=input_unit,
                description=f"The unit of the input token price.",
            )
            output_token_price: str = Field(
                default=output_price,
                description=f"The price per {output_unit} token (in {output_currency}).",
            )
            output_token_unit: str = Field(
                default=output_unit,
                description=f"The unit of the output token price.",
            )
            budget_mode: BudgetMode = Field(
                default=BudgetMode.DISABLED,
                description=(
                    "The budget mode for the model, which controls cost monitoring and enforcement. "
                    "Options:\n"
                    "Disabled: No budget tracking or restrictions.\n"
                    "Monitor: Logs the cost of each invocation without any notifications or enforcement.\n"
                    "Notify: Sends a warning notification when the budget limit is exceeded.\n"
                    "Trace: Appends cost breakdown details to the model’s response, including request cost and total usage.\n"
                    "Block: Prevents further invocations once the budget limit is exceeded, returning an error message instead."
                ),
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
def agent_prompt_prefix(prefix, cat):
    prefix = """Please do not include any cost breakdowns, request costs, or total cost information in responses. 
        Focus only on the main conversation topic and user requests."""
    return prefix


@tool(
    "Reset Cumulative Model Cost",
    return_direct=False,
    examples=[
        "Reset the stored model cost data.",
        "Clear all accumulated cost records for the model.",
        "Delete the current cost cache and start fresh.",
    ],
)
def reset_cached_model_costs(data, cat):
    """Reset the cumulative model cost data.

    This function clears the cached record of the total cost accumulated across all model invocations.
    It ensures that future cost tracking starts from zero.
    """
    try:
        with open(CACHED_COST_FILE, "w") as f:
            json.dump({}, f)
        return "✅ Cumulative model cost has been reset."
    except Exception as e:
        return f"❌ Error resetting cumulative model cost: {str(e)}"


# @tool(
#     "Reset Model Pricing Data",
#     return_direct=False,
#     examples=[
#         "Reset the token cost statistics for the model.",
#         "Clear all stored pricing information for LLM calls.",
#         "Delete the pricing cache and refresh it.",
#     ],
# )
# def reset_cached_model_pricing(data, cat):
#     """Reset token cost statistics for each LLM call.

#     This function clears the cached pricing data, which tracks the cost per token for different LLM calls.
#     After resetting, the system will need to reload or re-fetch pricing information.
#     """
#     try:
#         with open(CACHED_PRICING_FILE, "w") as f:
#             json.dump({}, f)
#         return "✅ Token cost statistics have been reset."
#     except Exception as e:
#         return f"❌ Error resetting token cost statistics: {str(e)}"


@tool(
    "Get Current Model Cost",
    return_direct=False,
    examples=[
        "What is the total cost of my model usage?",
        "Show me the current cumulative cost for the model.",
        "How much have I spent on LLM calls so far?",
    ],
)
def get_current_model_cost(data, cat):
    """Retrieve the current cumulative model cost.

    Reads the cached model cost data and returns the total cost accumulated across all model invocations.
    """
    try:
        if not os.path.exists(CACHED_COST_FILE):
            return "⚠️ No cost data found. The cache might be empty."

        with open(CACHED_COST_FILE, "r") as f:
            cost_data = json.load(f)

        total_cost = cost_data.get("current_cost", 0.0)

        return (
            f"💰 **Total Accumulated Cost:** `${total_cost:.6f}`\n"
            "🔹 This includes all previous model invocations.\n"
            "⚠️ *The cost of the current request is not included and will be added after execution.*"
        )
    except Exception as e:
        return f"❌ Error retrieving model cost: {str(e)}"


@tool(
    "Get Current Model Pricing",
    return_direct=False,
    examples=[
        "How much does my current model charge per token?",
        "What is the pricing for each model call?",
        "Show the token price for my LLM model.",
    ],
)
def get_current_model_pricing(data, cat):
    """Retrieve the pricing information for the current model.

    Reads the cached pricing data and returns the cost per token for the model in use.
    """
    try:
        model_class_name = crud.get_setting_by_name("llm_selected")["value"]["name"]
        model_arn = crud.get_setting_by_name(model_class_name)["value"]["model_id"]
        model_id = model_arn.split("/")[-1]

        if not os.path.exists(CACHED_PRICING_FILE):
            return "⚠️ No pricing data found. The cache might be empty."

        with open(CACHED_PRICING_FILE, "r") as f:
            pricing_data = json.load(f)

        model_pricing = pricing_data.get(model_arn, {}).get("data", {}).get(model_id)
        if not model_pricing:
            return f"⚠️ No pricing information available for model `{model_arn}`."

        input_price = model_pricing.get("input", {}).get("price")
        input_unit = model_pricing.get("input", {}).get("unit", 1000)
        output_price = model_pricing.get("output", {}).get("price")
        output_unit = model_pricing.get("output", {}).get("unit", 1000)

        input_price_str = (
            f"${float(input_price):.6f}"
            if isinstance(input_price, (int, float))
            else "N/A"
        )
        output_price_str = (
            f"${float(output_price):.6f}"
            if isinstance(output_price, (int, float))
            else "N/A"
        )

        return (
            f"💲 **Current Model Pricing for `{model_arn}`**\n"
            f"🔹 **Input Cost:** {input_price_str} per {input_unit} tokens\n"
            f"🔹 **Output Cost:** {output_price_str} per {output_unit} tokens"
        )

    except Exception as e:
        return f"❌ **Error retrieving model pricing:** {str(e)}"


@tool(
    "Get Current Model",
    return_direct=False,
    examples=[
        "Which LLM model am I using?",
        "What is my current AI model?",
        "Show the model name I am working with.",
    ],
)
def get_current_model(data, cat):
    """Retrieve the currently selected AI model.

    Returns the name and ARN of the model in use.
    """
    try:
        model_class_name = crud.get_setting_by_name("llm_selected")["value"]["name"]
        model_arn = crud.get_setting_by_name(model_class_name)["value"]["model_id"]

        return (
            f"🤖 **Current Model Information**\n"
            f"🔹 **Model Class Name:** `{model_class_name}`\n"
            f"🔹 **Model Amazon Resource Name:** `{model_arn}`"
            "You can use this information to identify the model and its capabilities."
        )

    except Exception as e:
        return f"❌ **Error retrieving model information:** {str(e)}"


@hook
def factory_allowed_llms(allowed, cat) -> List:
    return allowed + factory_pipeline()
