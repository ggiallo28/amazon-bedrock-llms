import json
import logging
from typing import Dict, List, Any
# from json_repair import repair_json
import botocore.exceptions
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
DEFAULT_REGION = "us-east-1"


def get_aws_service_pricing(
    service_code: str, client: Any, model_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve and filter AWS service pricing information.

    Args:
        service_code (str): The AWS service code.
        client (Any): The boto3 pricing client.
        model_id (str): The model ID to filter pricing information.
        region (str, optional): The AWS region. Defaults to DEFAULT_REGION.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Filtered pricing information.
    """
    model_parts = model_id.replace("-", ".").split(".")

    try:
        response = client.get_products(
            ServiceCode=service_code,
            Filters=[
                {
                    "Type": "TERM_MATCH",
                    "Field": "regionCode",
                    "Value": client.meta.region_name,
                },
                {
                    "Type": "TERM_MATCH",
                    "Field": "feature",
                    "Value": "On-demand Inference",
                },
            ],
            MaxResults=100,
        )
    except Exception as e:
        logger.error(f"Error fetching AWS service pricing: {e}")
        return {"PriceList": []}

    filtered_price_list = []
    for price_item in response.get("PriceList", []):
        try:
            price_data = json.loads(price_item)
            product_attributes = price_data.get("product", {}).get("attributes", {})
            model_attribute = product_attributes.get("model", "").lower()

            if any(part.lower() in model_attribute for part in model_parts):
                for term in price_data.get("terms", {}).get("OnDemand", {}).values():
                    for price_dimension in term.get("priceDimensions", {}).values():
                        pricing_info = {
                            "unit": price_dimension.get("unit"),
                            "endRange": price_dimension.get("endRange"),
                            "description": price_dimension.get("description"),
                            "appliesTo": price_dimension.get("appliesTo"),
                            "rateCode": price_dimension.get("rateCode"),
                            "beginRange": price_dimension.get("beginRange"),
                            "pricePerUnit": price_dimension.get("pricePerUnit"),
                            "model": model_attribute,
                        }
                        filtered_price_list.append(pricing_info)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse price item: {price_item}")
            continue

    return {"PriceList": filtered_price_list}


def parse_pricing_with_model(
    content: str, model_id: str, client: Any
) -> Dict[str, Any]:
    """
    Parse pricing information using an AI model.

    Args:
        content (str): The pricing content to parse.
        model_id (str): The model ID to use for parsing.
        client (Any): The boto3 bedrock-runtime client.

    Returns:
        Dict[str, Any]: Parsed pricing information.
    """
    prompt = f"""
    You are an AI assistant tasked with extracting pricing information for the AWS Bedrock model with ID '{model_id}' from the following content. Please analyze the content and provide a JSON output with the pricing details for this specific model. The JSON should be structured as follows:
    
    {{
        "{model_id}": {{
            "input": {{
                "price": float,
                "unit": float,
                "currency": str
            }},
            "output": {{
                "price": float,
                "unit": float,
                "currency": str
            }}
        }}
    }}

    If the specific model ID is not found, please provide pricing information for the most similar model or the general category it belongs to.
    Here's the content:

    {content}
    
    Please provide only the JSON output without any additional explanation.
    """

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0, "topP": 1},
        )
        generated_text = response["output"]["message"]["content"][0]["text"]

        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse generated text, attempting repair")
            return json.loads(repair_json(generated_text))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            logger.error(
                f"Access denied for model {model_id}: {e.response['Error']['Message']}"
            )
            return {
                "error": "AccessDenied",
                "message": "You do not have permission to access this model.",
            }
        else:
            logger.error(f"Error in parse_pricing_with_model for {model_id}: {e}")
            return {"error": "ClientError", "message": str(e)}
    except Exception as e:
        logger.error(f"Error in parse_pricing_with_model: {e}")
        return {}


def main(model: str, region: str):
    """
    Main function to retrieve and parse AWS Bedrock pricing information.

    Args:
        model (str): The model ID to use.
        region (str): The AWS region to use.
    """
    import boto3

    try:
        #bedrock = boto3.client("bedrock-runtime", region_name=region)
        pricing = boto3.client("pricing", region_name=region)

        content = get_aws_service_pricing("AmazonBedrock", pricing, model)

        print(">>>", content)
        # pricing_data = parse_pricing_with_model(json.dumps(content), model, bedrock)

        # with open("bedrock_pricing.json", "w") as f:
        #     json.dump(pricing_data, f, indent=2)

        # logger.info(
        #     f"Pricing data for model {model} in region {region} has been parsed and saved to bedrock_pricing.json"
        # )
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
