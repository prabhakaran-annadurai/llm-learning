### LLM Optimization

1. Prompt Engineering:

    Based on question or instructions

    pros:

        -   Easy to implement
        -   Rapid prototyping

    cons:

        -   not for complex tasks


2. Fine tuning:

    Customize pretrained LLMs for a specific task

    - High quality data required
    - Domain experts needed


3. RAG:

    Integrate domain data with LLMs knowledge

    Langchain, LLamaIndex used for building RAG


Quantization:

    - Reduces storage required for running local models
    - 32 bit floating to 16 bit floats or 8 bit integers
    - Calculation:

            scale-factor = max integer value / max float value
            each weight * scale-factor

    - LLMs may have outlier weights (very large values)
    - LLM.int8() - regular weights & outlier weights processed separately
    - bitsandbytes library to quantize


Pruning and knowledge distillation:

    - Eliminating weights with little impact
    - Unstructured and Structured pruning(Removing entire layer)
    - Eg. SparseGPT, LLMPruner
    - Pruning + FineTuning - can be effective

    - Alpaca - 7B finetuned model
    - Black-box knowledge distillation and whitebox
    - MiniLLM


Hallucination:

    - False information
    - BERT score - calculates relevance between prompt and generated text - recall, precision


Detecting Data leakages:

    - string matching, regular expressions

Detecting Toxic content:

    - toxigen - library to evaulate input text


Prompt injection:

    - 