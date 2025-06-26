"""CAD-related service functions built on LangChain."""

import os
from typing import Dict, Any, Tuple

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from ..relay.vlm_client import call_vlm


def init_llm() -> OpenAI:
    """Initialise the LLM client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key, temperature=0.2)


def _build_chains(llm: OpenAI) -> SequentialChain:
    """Create the sequential LangChain pipeline."""
    step1_prompt = PromptTemplate(
        input_variables=["pdf", "json"],
        template=(
            "Step1: \u8bf7\u6839\u636e\u8f93\u5165PDF\u548cJSON\uff0c\u63d0\u53d6\u6240\u6709\u96f6\u4ef6\u5c3a\u5bf8\u4e0e\u7c7b\u578b..."
        ),
    )
    step2_prompt = PromptTemplate(
        input_variables=["step1_result"],
        template=(
            "Step2: \u4e0a\u4e00\u6b65\u63d0\u53d6\u7684\u7ed3\u679c\u5982\u4e0b\uff1a{step1_result}\u3002\u8bf7\u5206\u6790\u5404\u5c3a\u5bf8\u7684\u5408\u7406\u6027\uff0c\u5e76\u627e\u51fa\u7591\u70b9..."
        ),
    )
    step3_prompt = PromptTemplate(
        input_variables=["step2_result"],
        template=(
            "Step3: \u4f60\u7684\u4e0a\u4e00\u6b65\u5206\u6790\u5982\u4e0b\uff1a{step2_result}\u3002\u8bf7\u7ed3\u5408\u5de5\u7a0b\u89c4\u5219\u81ea\u67e5\u7ed3\u8bba\uff0c\u82e5\u53d1\u73b0\u9519\u8bef\u8bf7\u4fee\u6b63\u5e76\u8bf4\u660e\u601d\u8def..."
        ),
    )

    step1_chain = LLMChain(llm=llm, prompt=step1_prompt, output_key="step1_result")
    step2_chain = LLMChain(llm=llm, prompt=step2_prompt, output_key="step2_result")
    step3_chain = LLMChain(llm=llm, prompt=step3_prompt, output_key="analysis")

    return SequentialChain(
        chains=[step1_chain, step2_chain, step3_chain],
        input_variables=["pdf", "json"],
        output_variables=["analysis"],
    )


def _confirm_with_vlm(result: str, json_data: str) -> Dict[str, Any]:
    """Ask the VLM to validate LLM output against the CAD JSON."""
    payload = {"result": result, "json": json_data}
    return call_vlm(payload)


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze CAD data with a LangChain pipeline and VLM confirmation."""
    pdf_path = data.get("pdf")
    json_data = data.get("json")
    if not pdf_path or json_data is None:
        raise ValueError("'pdf' and 'json' fields are required")

    llm = init_llm()
    chain = _build_chains(llm)

    chain_result: Dict[str, Any] = chain({"pdf": pdf_path, "json": json_data})
    analysis = chain_result.get("analysis", "")

    confirmation = _confirm_with_vlm(analysis, json_data)

    return {"analysis": analysis, "confirmation": confirmation}

