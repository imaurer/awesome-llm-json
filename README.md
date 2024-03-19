# Awesome LLM JSON List

This awesome list is dedicated to resources for using Large Language Models (LLMs) to generate JSON or other structured outputs.
  
## Table of Contents  
  
* [Terminology](#terminology)  
* [Hosted Models](#hosted-models)
* [Local Models](#local-models)
* [Python Libraries](#python-libraries)
* [Blog Articles](#blog-articles)
* [Videos](#videos) 
* [Jupyter Notebooks](#jupyter-notebooks)
* [Leaderboards](#leaderboards)
  
## Terminology  
  
Unfortunately, generating JSON goes by a few different names that roughly mean the same thing:  
  
* Structured Outputs: Using an LLM to generate any structured output including JSON, XML, or YAML regardless of technique (e.g. function calling, guided generation).
* [Function Calling](https://www.promptingguide.ai/applications/function_calling): Providing an LLM a hypothetical (or actual) function definition for it to "call" in it's chat or completion response. The LLM doesn't actually call the function, it just provides an indication that one should be called via a JSON message.
* [JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode): Specifying that an LLM must generate valid JSON. Depending on the provider, a schema may or may not be specified and the LLM may create an unexpected schema.
* [Tool Usage](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools): Giving an LLM a choice of tools such as image generation, web search, and "function calling".  The functional calling parameter in the API request is now called "tools".
* [Guided Generation](https://arxiv.org/abs/2307.09702): For constraining an LLM model to generate text that follows a prescribed specification such as a [Context-Free Grammar](https://en.wikipedia.org/wiki/Context-free_grammar).
* [GPT Actions](https://platform.openai.com/docs/actions/introduction): ChatGPT invokes actions (i.e. API calls) based on the endpoints and parameters specified in an [OpenAPI specification](https://swagger.io/specification/). Unlike the capability called "Function Calling", this capability will indeed call your function hosted by an API server.

None of these names are great, that's why I named this list just "Awesome LLM JSON".
  
## Hosted Models

| Provider     | Models                                                                     | Links                                                                                                                                                                                                                                                                                                                                                                                                                        |
|--------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AnyScale     | Mistral-7B-Instruct-v0.1 Mixtral-8x7B-Instruct-v0.1                        | [Function Calling](https://docs.endpoints.anyscale.com/text-generation/function-calling) [JSON Mode](https://docs.endpoints.anyscale.com/text-generation/json-mode) [Pricing](https://docs.endpoints.anyscale.com/pricing/) [Announcement (2023)](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features)                                                                                  |
| Azure        | gpt-4 gpt-4-turbo gpt-35-turbo mistral-large-latest mistral-large-2402     | [Function Calling](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python) [OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing) [Mistral Pricing](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/000-000.mistral-ai-large-offer?tab=PlansAndPrice)                                                          |
| Anthropic    | Claude 2.1 Claude 3 Opus Claude 3 Sonnet Claude 3 Haiku                    | [Announcement 2023-21-11](https://www.anthropic.com/news/claude-2-1) [Docs](https://docs.anthropic.com/claude/docs/functions-external-tools) [Jupyter Notebook](https://github.com/anthropics/anthropic-cookbook/blob/main/function_calling/function_calling.ipynb)                                                                                                                                                          |
| Cohere       | Command-R                                                                  | [Function Calling](https://docs.cohere.com/docs/tool-use) [Pricing](https://cohere.com/pricing) [Announcement (2024-03-11)](https://txt.cohere.com/command-r/)                                                                                                                                                                                                                                                               |
| Fireworks.ai | firefunction-v1                                                            | [Function Calling](https://readme.fireworks.ai/docs/function-calling) [JSON Mode](https://readme.fireworks.ai/docs/structured-response-formatting) [Grammar mode](https://readme.fireworks.ai/docs/structured-output-grammar-based) [Pricing](https://fireworks.ai/pricing) [Announcement (2023-12-20)](https://blog.fireworks.ai/fireworks-raises-the-quality-bar-with-function-calling-model-and-api-release-e7f49d1e98e9) |
| Google       | gemini-1.0-pro                                                             | [Function Calling](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling#rest) [Pricing](https://ai.google.dev/pricing?authuser=1)                                                                                                                                                                                                                                                               |
| Mistral      | mistral-large-latest                                                       | [Function Calling](https://docs.mistral.ai/guides/function-calling/) [Pricing](https://docs.mistral.ai/platform/pricing/)                                                                                                                                                                                                                                                                                                    |
| OpenAI       | gpt-4 gpt-4-turbo gpt-35-turbo                                             | [Function Calling](https://openai.com/blog/openai-api/) [JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode) [Pricing](https://openai.com/pricing) [Announcement (2023-06-13)](https://openai.com/blog/function-calling-and-other-api-updates)                                                                                                                                                     |
| Rysana       | inversion-sm                                                               | [API Docs](https://rysana.com/docs/api) [Pricing](https://rysana.com/pricing) [Announcement (2024-03-18)](https://rysana.com/inversion)                                                                                                                                                                                                                                                                                      |
| Together AI  | Mixtral-8x7B-Instruct-v0.1 Mistral-7B-Instruct-v0.1 CodeLlama-34b-Instruct | [Function Calling](https://docs.together.ai/docs/function-calling) [JSON Mode](https://docs.together.ai/docs/json-mode) [Pricing](https://together.ai/pricing/) [Announcement 2024-01-31](https://www.together.ai/blog/function-calling-json-mode)                                                                                                                                                                           |

**Parallel Function Calling**

Below is a list of hosted API models that support multiple parallel function calls. This could include checking the weather in multiple cities or first finding the location of a hotel and then checking the weather at it's location.

- azure/openai
	- gpt-4-turbo-preview
	- gpt-4-1106-preview
	- gpt-4-0125-preview
	- gpt-3.5-turbo-1106
	- gpt-3.5-turbo-0125
- cohere
	- command-r
- together_ai
	- Mixtral-8x7B-Instruct-v0.1
	- Mistral-7B-Instruct-v0.1
	- CodeLlama-34b-Instruct
 
## Local Models

[Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) (2024-03-13, Nous Research) is a 7B parameter model that excels at function calling, JSON structured outputs, and general tasks. Trained on an updated OpenHermes 2.5 Dataset and a new function calling dataset, it uses a special system prompt and multi-turn structure. Achieves 91% on function calling and 84% on JSON mode evaluations.

[Gorilla OpenFunctions v2](https://gorilla.cs.berkeley.edu//blogs/7_open_functions_v2.html) (2024-02-27, Apache 2.0 license, [Charlie Cheng-Jie Ji et al.](https://gorilla.cs.berkeley.edu//blogs/7_open_functions_v2.html))  interprets and executes functions based on JSON Schema Objects, supporting multiple languages and detecting function relevance.

[NexusRaven-V2](https://nexusflow.ai/blogs/ravenv2) (2023-12-05, Nexusflow)  is a 13B model outperforming GPT-4 in zero-shot function calling by up to 7%, enabling effective use of software tools. Further instruction-tuned on CodeLlama-13B-instruct.

[Functionary](https://functionary.meetkai.com/) (2023-08-04, [MeetKai](https://meetkai.com/)) interprets and executes functions based on JSON Schema Objects, supporting various compute requirements and call types. Compatible with OpenAI-python and llama-cpp-python for efficient function execution in JSON generation tasks.

## Python Libraries


[DSPy](https://github.com/stanfordnlp/dspy) (MIT) is a framework for algorithmically optimizing LM prompts and weights. DSPy introduced [typed predictor and signatures](https://github.com/entropy/dspy/blob/main/docs/docs/building-blocks/8-typed_predictors.md) to leverage [Pydantic](https://github.com/pydantic/pydantic) for enforcing type constraints on inputs and outputs, improving upon string-based fields. 

[FuzzTypes](https://github.com/genomoncology/FuzzTypes) (MIT) extends Pydantic with autocorrecting annotation types for enhanced data normalization and handling of complex types like emails, dates, and custom entities.

[guidance](https://github.com/guidance-ai/guidance) (Apache-2.0) enables constrained generation, interleaving Python logic with LLM calls, reusable functions, and calling external tools. Optimizes prompts for faster generation.

[Instructor](https://github.com/jxnl/instructor) (MIT) simplifies generating structured data from LLMs using Function Calling, Tool Calling, and constrained sampling modes. Built on Pydantic for validation and supports various LLMs.

[LangChain](https://github.com/langchain-ai/langchain) (MIT) provides an interface for chains, integrations with other tools, and chains for applications. LangChain offers [chains for structured outputs](https://python.langchain.com/docs/modules/chains/how_to/structured_outputs) and [function calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling) across models.

[LiteLLM](https://github.com/BerriAI/litellm) (MIT) simplifies calling 100+ LLMs in the OpenAI format, supporting [function calling](https://docs.litellm.ai/docs/completion/function_call), tool calling, and JSON mode.

[LlamaIndex](https://github.com/run-llama/llama_index) (MIT) provides [modules for structured outputs](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/structured_outputs.html) at different levels of abstraction, including output parsers for text completion endpoints, [Pydantic programs](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/pydantic_program.html) for mapping prompts to structured outputs using function calling or output parsing, and pre-defined Pydantic programs for specific output types.

[Marvin](https://github.com/PrefectHQ/marvin) (Apache-2.0) is a lightweight toolkit for building reliable natural language interfaces with self-documenting tools for tasks like entity extraction and multi-modal support.

[Outlines](https://github.com/outlines-dev/outlines) (Apache-2.0) facilitates structured text generation using multiple models, Jinja templating, and support for regex patterns, JSON schemas, Pydantic models, and context-free grammars.

[Pydantic](https://github.com/pydantic/pydantic) (MIT) simplifies working with data structures and JSON through data model definition, validation, JSON schema generation, and seamless parsing and serialization.

[SGLang](https://github.com/sgl-project/sglang) (MPL-2.0) allows specifying JSON schemas using regular expressions or Pydantic models for constrained decoding. Its high-performance runtime accelerates JSON decoding.

## Blog Articles

[Structured Generation Improves LLM performance: GSM8K Benchmark](https://blog.dottxt.co/performance-gsm8k.html) (2024-03-15, .txt Engineering) demonstrates consistent improvements across 8 models, highlighting benefits like "prompt consistency" and "thought-control."

[LoRAX + Outlines: Better JSON Extraction with Structured Generation and LoRA](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) (2024-03-03, Predibase Blog) combines Outlines with LoRAX v0.8 to enhance extraction accuracy and schema fidelity through structured generation, fine-tuning, and LoRA adapters.

[FU, Show Me The Prompt. Quickly understand inscrutable LLM frameworks by intercepting API calls](https://hamel.dev/blog/posts/prompt/) (2023-02-14, Hamel Husain) provides a practical guide to intercepting API calls using mitmproxy, gaining insights into tool functionality, and assessing necessity. Emphasizes minimizing complexity and maintaining close connection with underlying LLMs.

[Coalescence: making LLM inference 5x faster](https://blog.dottxt.co/coalescence.html) (2024-02-02, .txt Engineering) shows how structured generation can be made faster than unstructured generation using a technique called "coalescence", with a caveat regarding how it may affect the quality of the generation.

[Getting Started with Function Calling](https://www.promptingguide.ai/applications/function_calling) (2024-01-11, Elvis Saravia) introduces function calling for connecting LLMs with external tools and APIs, providing an example using OpenAI's API and highlighting potential applications.

[Pushing ChatGPT's Structured Data Support To Its Limits](https://minimaxir.com/2023/12/chatgpt-structured-data/) (2023-12-21, Max Woolf) delves into leveraging ChatGPT's capabilities using paid API, JSON schemas, and Pydantic. Highlights techniques for improving output quality and the benefits of structured data support.

[Why use Instructor?](https://jxnl.github.io/instructor/why/) (2023-11-18, Jason Liu) explains the benefits of the library, offering a readable approach, support for partial extraction and various types, and a self-correcting mechanism. Recommends additional resources on the Instructor website.

[Using grammars to constrain llama.cpp output](https://www.imaurer.com/llama-cpp-grammars/) (2023-09-06, Ian Maurer) integrates context-free grammars with llama.cpp for more accurate and schema-compliant responses, particularly for biomedical data.

[Using OpenAI functions and their Python library for data extraction](https://til.simonwillison.net/gpt3/openai-python-functions-data-extraction) (2023-07-09, Simon Willison) demonstrates extracting structured data using OpenAI Python library and function calling in a single API call, with a code example and suggestions for handling streaming limitations.

## Videos

[Hermes 2 Pro Overview](https://www.youtube.com/watch?v=ViXURxck-HM) (2024-03-18, Prompt Engineer)  introduces Hermes 2 Pro, a 7B parameter model excelling at function calling and structured JSON output. Demonstrates 90% accuracy in function calling and 84% in JSON mode, outperforming other models.

[Mistral AI Function Calling](https://www.youtube.com/watch?v=eOo4GfHj3ZE) (2024-02-24, Sophia Yang) demonstrates connecting LLMs to external tools, generating function arguments, and executing functions. Could be extended to generate or manipulate JSON data.

[Function Calling in Ollama vs OpenAI](https://www.youtube.com/watch?v=RXDWkiuXtG0)  (2024-02-13, [Matt Williams](https://twitter.com/Technovangelist)) clarifies that models generate structured output for parsing and invoking functions. Compares implementations, highlighting Ollama's simpler approach and using few-shot prompts for consistency.

[LLM Engineering: Structured Outputs](https://www.youtube.com/watch?v=1xUeL63ymM0) (2024-02-12, [Jason Liu](https://twitter.com/jxnlco), [Weights & Biases Course](https://www.wandb.courses/)) offers a concise course on handling structured JSON output, function calling, and validations using Pydantic, covering essentials for robust pipelines and efficient production integration.

[Why Pydantic became indispensable for LLMs](https://www.factsmachine.ai/p/how-pydantic-became-indispensable) (2024-01-19, [Adam Azzam](https://twitter.com/aaazzam)) explains Pydantic's emergence as a critical tool, enabling sharing data models via JSON schemas and reasoning between unstructured and structured data. Highlights the importance of quantizing the decision space and potential issues with LLMs overfitting to older schema versions.

[Pydantic is all you need](https://www.youtube.com/watch?v=yj-wSRJwrrc)  (2023-10-10, [Jason Liu](https://twitter.com/jxnlco), [AI Engineer Conference](https://www.ai.engineer/)) discusses the importance of Pydantic for structured prompting and output validation, introducing the Instructor library and showcasing advanced applications for reliable and maintainable LLM-powered applications.

## Jupyter Notebooks

[Function Calling with llama-cpp-python and OpenAI Python Client](https://github.com/abetlen/llama-cpp-python/blob/main/examples/notebooks/Functions.ipynb) demonstrates integration, including setup using the Instructor library, with examples of retrieving weather information and extracting user details.

[Function Calling with Mistral Models](https://colab.research.google.com/github/mistralai/cookbook/blob/main/function_calling.ipynb) demonstrates connecting Mistral models with external tools through a simple example involving a payment transactions dataframe.

[chatgpt-structured-data](https://github.com/minimaxir/chatgpt-structured-data) by [Max Woolf](https://twitter.com/minimaxir) provides demos showcasing ChatGPT's function calling and structured data support, covering various use cases and schemas.## Leaderboards

[Function Calling with Claude](https://github.com/anthropics/anthropic-cookbook/blob/main/function_calling/function_calling.ipynb) by Anthropic 



## Leaderboards

[Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) is an evaluation framework for LLMs' function-calling capabilities including over 2k question-function-answer pairs across languages like Python, Java, JavaScript, SQL, and REST API, focusing on simple, multiple, and parallel function calls, as well as function relevance detection.
