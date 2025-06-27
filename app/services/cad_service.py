
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
            "Step1: 请根据输入PDF和JSON，提取所有零件尺寸与类别..."
        ),
    )
    step2_prompt = PromptTemplate(
        input_variables=["step1_result"],
        template=(
            "Step2: 上一步提取的结果如下：{step1_result}。请分析各尺寸的合理性，并找出疑点..."
        ),
    )
    step3_prompt = PromptTemplate(
        input_variables=["step2_result"],
        template=(
            "Step3: 你的上一步分析如下：{step2_result}。请结合工程规则自查结论，若发现错误请修正并说明思路..."
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

