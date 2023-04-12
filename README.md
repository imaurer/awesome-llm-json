# awesome-decentralized-llm

Collection of LLM resources that can be used to build products you can "own" or to perform reproducible research. Please note there are Terms of Service around some of the weights and training data that should be investigated before commercialization.

Currently collecting information on [Autonomous Agents](#autonomous-agents) and [Edge LLMs](#edge-llms), but will add new sections as the field evolves.

-----

## Autonomous Agents


### Autonomous Agent Repositories

- [babyagi](https://github.com/yoheinakajima/babyagi) -
  Python script example of AI-powered task management system. Uses OpenAI and Pinecone APIs to create, prioritize, and execute tasks. 
  (2023-04-06, Yohei Nakajima)

- [Auto-GPT](https://github.com/Torantulino/Auto-GPT) -
  An experimental open-source attempt to make GPT-4 fully autonomous.
  (2023-04-06, Toran Bruce Richards)

- [JARVIS](https://github.com/microsoft/JARVIS) -
  JARVIS, a system to connect LLMs with ML community
  (2023-04-06, Microsoft)


### Autonomous Agent Resources

- [Emergent autonomous scientific research capabilities of large language models](https://arxiv.org/abs/2304.05332)
  (2023-04-11, Daniil A. Boiko,1 Robert MacKnight, and Gabe Gomes - Carnegie Mellon University)

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf)
  (2023-04-07, Stanford and Google)

- [Twitter List: Homebrew AGI Club](https://twitter.com/i/lists/1642934512836575232)
  (2023-04-06, [@altryne](https://twitter.com/altryne)]

- [LangChain: Custom Agents](https://blog.langchain.dev/custom-agents/)
  (2023-04-03, LangChain)
 
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)
  (2023-04-02, Microsoft)

- [Introducing "🤖 Task-driven Autonomous Agent"](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)
  (2023-03-29, [@yoheinakajima](https://twitter.com/yoheinakajima))

- [A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern)
  (2023-03-17, Simon Willison)

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io/)
  (2023-03-10, Princeton & Google)


-----

## Edge LLMs

### Edge LLM Repositories

- [Basaran](https://github.com/hyperonym/basaran)
  Open-source text completion API for Transformers-based text generation models.
  (2023-04-12, Hyperonym)

- [TurboPilot](https://github.com/ravenscroftj/turbopilot)
  CoPilot clone that runs code completion 6B-LLM with CPU and 4GB of RAM.
  (2023-04-11, James Ravenscroft)
  
- [LMFlow](https://github.com/OptimalScale/LMFlow)
  An Extensible Toolkit for Finetuning and Inference of Large Foundation Models.
  (2023-04-06, OptimalScale)

- [xturing](https://github.com/stochasticai/xturing) -
  Build and control your own LLMs
  (2023-04-03, stochastic.ai)

- [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) -
  4 bits quantization of LLaMA using GPTQ
  (2023-04-01, qwopqwop200, Meta ToS)

- [GPT4All](https://github.com/nomic-ai/gpt4all) -
  LLM trained with ~800k GPT-3.5-Turbo Generations based on LLaMa.
  (2023-03-28, Nomic AI, OpenAI ToS)
 
- [Dolly](https://github.com/databrickslabs/dolly) -
  Large language model trained on the Databricks Machine Learning Platform
  (2023-03-24, Databricks Labs, Apache)
  
- [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp)
  Inference of HuggingFace's BLOOM-like models in pure C/C++.
  (2023-03-16, Nouamane Tazi, MIT License)
  
- [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp) -
  Locally run an Instruction-Tuned Chat-Style LLM
  (2023-03-16, Kevin Kwok, MIT License)

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) -
  Code and documentation to train Stanford's Alpaca models, and generate the data.
  (2023-03-13, Stanford CRFM, Apache License, Non-Commercial Data, Meta/OpenAI ToS)

- [llama.cpp](https://github.com/ggerganov/llama.cpp) -
  Port of Facebook's LLaMA model in C/C++. 
  (2023-03-10, Georgi Gerganov, MIT License)

- [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) -
  ChatRWKV is like ChatGPT but powered by RWKV (100% RNN) language model, and open source.
  (2023-01-09, PENG Bo, Apache License)
  
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) -
  RNN with Transformer-level LLM performance. Combines best of RNN and transformer: fast inference, saves VRAM, fast training.
  (2022?, PENG Bo, Apache License)


### Edge LLM Spaces, Models & Datasets

- [Dolly 15k Instruction Tuning Labels](https://github.com/databrickslabs/dolly/tree/master/data)
  (2023-04-12, DataBricks, CC3 Allows Commercial Use)
  
- [Cerebras-GPT 7 Models](https://huggingface.co/cerebras)
  (2023-03-28, Huggingface, Cerebras, Apache License)

- [Alpine Data Cleaned](https://github.com/gururise/AlpacaDataCleaned)
  (2023-03-21, Gene Ruebsamen, Apache & OpenAI ToS)

- [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
  (2023-03-13, Huggingface, Tatsu-Lab, Meta ToS/OpenAI ToS)
  
- [Alpaca Model Search](https://huggingface.co/models?sort=downloads&search=alpaca)
  (Huggingface, Meta ToS/OpenAI ToS)
  

### Edge LLM Resources

- [Summary of Curent Models](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs/edit#gid=1158069878)
  (2023-04-11, Dr Alan D. Thompson, Google Sheet)

- [Running GPT4All On a Mac Using Python langchain in a Jupyter Notebook](https://blog.ouseful.info/2023/04/04/running-gpt4all-on-a-mac-using-python-langchain-in-a-jupyter-notebook/)
  (2023-04-04, Tony Hirst, Blog Post)

- [Cerebras-GPT vs LLaMA AI Model Comparison](https://www.lunasec.io/docs/blog/cerebras-gpt-vs-llama-ai-model-comparison/)
  (2023-03-29, LunaSec, Blog Post)

- [Cerebras-GPT: Family of Open, Compute-efficient, LLMs](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
  (2023-03-28, Cerebras, Blog Post)

- [Hello Dolly: Democratizing the magic of ChatGPT with open models](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
  (2023-03-24, databricks, Blog Post)

- [The Coming of Local LLMs](https://nickarner.com/notes/the-coming-of-local-llms-march-23-2023/)
  (2023-03-23, Nick Arner, Blog Post)

- [The RWKV language model: An RNN with the advantages of a transformer](https://johanwind.github.io/2023/03/23/rwkv_overview.html)
  (2023-03-23, Johan Sokrates Wind, Blog Post)
  
- [Bringing Whisper and LLaMA to the masses](https://changelog.com/podcast/532)
  (2023-03-15, The Changelog & Georgi Gerganov, Podcast Episode)
  
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  (2023-03-13, Stanford CRFM, Project Homepage)

- [Large language models are having their Stable Diffusion moment](https://simonwillison.net/2023/Mar/11/llama/)
  (2023-03-10, Simon Willison, Blog Post)

- [Running LLaMA 7B and 13B on a 64GB M2 MacBook Pro with llama.cpp](https://til.simonwillison.net/llms/llama-7b-m2)
  (2023-03-10, Simon Willison, Blog/Today I Learned)
  
- [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
  (2023-02-24, Meta AI, Meta ToS)
