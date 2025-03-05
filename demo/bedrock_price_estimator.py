import json
import logging
from typing import Dict, List, Any
import botocore.exceptions
import argparse
import re
import json
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_REGION = "us-east-1"


def fetch_aws_pricing(client):
    """
    Fetch AWS Bedrock pricing for a specific provider, handling pagination.

    Parameters:
        client (boto3.client): AWS Pricing client.
        provider_name (str): The provider name to filter (e.g., "anthropic", "mistral").

    Returns:
        dict: A dictionary containing the filtered PriceList.
    """
    all_price_list = []
    next_token = None

    try:
        while True:
            request_params = {
                "ServiceCode": "AmazonBedrock",
                "Filters": [
                    {
                        "Type": "TERM_MATCH",
                        "Field": "regionCode",
                        "Value": client.meta.region_name,
                    }
                ],
                "MaxResults": 100,
            }

            if next_token:
                request_params["NextToken"] = next_token

            response = client.get_products(**request_params)

            price_list = response.get("PriceList", [])
            all_price_list.extend(json.loads(price) for price in price_list)

            next_token = response.get("NextToken")

            if not next_token:
                break

        return all_price_list

    except Exception as e:
        logger.error(f"Error fetching AWS service pricing: {e}")
        return []


def get_model_names(prices: List[Dict[str, Any]]) -> List[str]:
    model_names = set()

    for price in prices:
        try:
            model_name = price["product"]["attributes"].get("model")
            if model_name:
                model_names.add(model_name)
        except KeyError:
            continue

    return sorted(model_names)


def parse_pricing_with_model(model_names: str, model_id: str, client: Any) -> str:
    prompt = f"""
    You are an AI assistant tasked with identifying the correct model name for the AWS Bedrock model with ID '{model_id}'.
    Below is a list of available model names: {", ".join(model_names)}

    Your task is to determine which model name from the list corresponds to the given model ID.
    If there is no exact match, return the closest matching model.

    **Output Format:**  
    - Only return the model name as a single string.  
    - Do NOT include any explanations, JSON, or additional text.
    """

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 50, "temperature": 0, "topP": 1},
        )
        model_name = response["output"]["message"]["content"][0]["text"].strip()
        return model_name

    except botocore.exceptions.ClientError as e:
        logger.error(f"Error fetching model name for {model_id}: {e}")
        return "Error"

    except Exception as e:
        logger.error(f"Unexpected error in parse_pricing_with_model: {e}")
        return "Error"


def filter_pricing_by_model(
    pricing_data: List[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    """
    Filters pricing data to include only entries that match the specified model name.

    Args:
        pricing_data (List[Dict[str, Any]]): List of pricing information.
        model_name (str): The model name to filter by.

    Returns:
        List[Dict[str, Any]]: Filtered list of pricing data for the given model.
    """
    filtered_pricing = []

    for entry in pricing_data:
        try:
            product_info = entry.get("product", {})
            attributes = product_info.get("attributes", {})

            if attributes.get("model") == model_name:
                filtered_pricing.append(entry)

        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping entry due to error: {e}")
            continue

    return filtered_pricing


def extract_model_pricing(
    filtered_pricing: List[Dict[str, Any]], model_id: str
) -> Dict[str, Any]:
    """
    Extracts pricing details for the given model from the filtered pricing data.

    Args:
        filtered_pricing (List[Dict[str, Any]]): Filtered list of pricing data for the model.
        model_id (str): The AWS Bedrock model ID.

    Returns:
        Dict[str, Any]: A dictionary containing pricing details for input, output, and cache read input tokens.
    """
    pricing_details = {"input": None, "output": None, "cache_read_input": None}

    found_types = set()  # Track which inference types are found

    for entry in filtered_pricing:
        try:
            product_info = entry.get("product", {})
            attributes = product_info.get("attributes", {})

            inference_type = attributes.get("inferenceType", "").lower()
            found_types.add(inference_type)

            on_demand_terms = entry.get("terms", {}).get("OnDemand", {})
            for term in on_demand_terms.values():
                price_dimensions = term.get("priceDimensions", {})
                for dimension in price_dimensions.values():
                    try:
                        price_per_unit = float(dimension["pricePerUnit"]["USD"])
                        unit = dimension.get("unit", "unknown")
                        currency = list(dimension["pricePerUnit"].keys())[0]

                        if (
                            "input tokens" in inference_type
                            and "cache read" not in inference_type
                        ):
                            pricing_details["input"] = {
                                "price": price_per_unit,
                                "unit": unit,
                                "currency": currency,
                            }
                        elif "output tokens" in inference_type:
                            pricing_details["output"] = {
                                "price": price_per_unit,
                                "unit": unit,
                                "currency": currency,
                            }
                        elif (
                            "cache read input tokens" in inference_type
                            or "prompt cache read input tokens" in inference_type
                        ):
                            pricing_details["cache_read_input"] = {
                                "price": price_per_unit,
                                "unit": unit,
                                "currency": currency,
                            }
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error parsing pricing details: {e}")
                        continue

        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping entry due to error: {e}")
            continue

    expected_types = {"input tokens", "output tokens", "cache read input tokens"}
    missing_types = expected_types - found_types

    if missing_types:
        logger.warning(f"Missing inference types for model {model_id}: {missing_types}")

    return {model_id: pricing_details}


def main(model: str, region: str):
    """
    Main function to retrieve and parse AWS Bedrock pricing information.

    Args:
        model (str): The model ID to use.
        region (str): The AWS region to use.
    """
    import boto3

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        pricing = boto3.client("pricing", region_name=region)
        prices = fetch_aws_pricing(pricing)

        model_names = get_model_names(prices)
        current_model_name = parse_pricing_with_model(model_names, model, bedrock)
        print(f"{model} = {current_model_name}")
        filtered_prices = filter_pricing_by_model(prices, current_model_name)
        final = extract_model_pricing(filtered_prices, model)

        print(final)
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve and parse AWS Bedrock pricing information."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="The model ID to use"
    )
    parser.add_argument(
        "--region", type=str, default=DEFAULT_REGION, help="The AWS region to use"
    )
    args = parser.parse_args()

    main(args.model, args.region)
