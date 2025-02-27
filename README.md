# Amazon Bedrock LLMs

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

This plugin integrates Amazon Bedrock LLMs (Language Models) into the Cheshire Cat AI framework, allowing users to leverage a variety of powerful language models provided by Amazon Web Services (AWS).

## Key Features

- Dynamic model selection: Automatically fetches and configures available Bedrock models.
- Guardrail support: Implements AWS Bedrock guardrails for enhanced security and compliance.
- Streaming support: Enables streaming responses for supported models.
- Flexible configuration: Allows customization of model parameters and settings.

## How It Works

1. The plugin uses the AWS Boto3 client to interact with Amazon Bedrock services.
2. It dynamically fetches available models and guardrails from the Bedrock API.
3. Custom Bedrock LLM classes are created for each available model.
4. The plugin integrates with the Cheshire Cat AI framework, allowing the use of these models in various AI tasks.

## Configuration

The plugin provides a dynamic settings model that allows users to:

- Enable/disable specific Bedrock models
- Configure model-specific parameters (e.g., temperature, max tokens)
- Set guardrails for enhanced security and compliance
- Customize model behavior through additional keyword arguments

## Usage

1. Ensure you have the necessary AWS credentials and permissions to access Amazon Bedrock services.
2. Install the plugin in your Cheshire Cat AI environment.
3. Configure the desired Bedrock models and settings through the Cheshire Cat AI interface.
4. The plugin will automatically integrate the selected Bedrock models into your AI pipeline.

For detailed configuration options and advanced usage, please refer to the plugin settings in the Cheshire Cat AI interface.

## Note

This plugin requires an active AWS account with access to Amazon Bedrock services. Make sure you understand the pricing and usage terms of Amazon Bedrock before using this plugin in production environments.

