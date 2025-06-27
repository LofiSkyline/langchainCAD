"""CAD-related service functions built on LangChain."""

import os
from typing import Dict, Any

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
    """Create the sequential LangChain pipeline for CAD analysis."""

    structure_prompt = PromptTemplate(
        input_variables=["pdf_data", "json_data"],
        template=
        """\
\u6839\u636e\u6587\u4ef6\u4e2d\u7684\u6570\u636e\u5bf9\u96f6\u4ef6\u8fdb\u884c\u7ed3\u6784\u5de5\u827a\u6027\u5206\u6790\uff0c\u9700\u8981\u5206\u6790\uff1a\n1. \u6750\u6599\u7684\u9009\u62e9\u548c\u70ed\u5904\u7406\u5de5\u827a\n2. \u5c3a\u5bf8\u5206\u6790\u7cbe\u5ea6\u7b49\u7ea7\u4ee5\u53ca\u52a0\u5de5\u65b9\u6cd5\u68c0\u6d4b\u65b9\u6cd5\n3. \u5f62\u4f4d\u516c\u5dee\u7cbe\u5ea6\u7b49\u7ea7\u4ee5\u53ca\u52a0\u5de5\u65b9\u6cd5\u68c0\u6d4b\u65b9\u6cd5\n4. \u8868\u9762\u7c97\u7ec6\u5ea6\u52a0\u5de5\u65b9\u6cd5\n5. \u7279\u6b8a\u5de5\u827a\u8981\u6c42\n\n\u516c\u5dee\u5c3a\u5bf8\u6570\u636e\u6587\u6863\uff1a\n{json_data}\n\n\u8bf7\u5bf9\u6bcf\u4e00\u4e2a\u5c3a\u5bf8\u8fdb\u884c\u8be6\u7ec6\u5206\u6790\uff0c\u786e\u4fdd\u5305\u542b\u5168\u90e8\u5c3a\u5bf8\u3002\n\u8bf7\u4f7f\u7528 Markdown \u683c\u5f0f\u8f93\u51fa\uff0c\u5305\u62ec\uff1a\n- \u4f7f\u7528\u6807\u9898\u5c42\u7ea7(#\u3001##\u3001###)\u7ec4\u7ec7\u5185\u5bb9\n- \u4f7f\u7528\u8868\u683c\u5c55\u793a\u6570\u636e\n- \u4f7f\u7528\u5217\u8868\u5c55\u793a\u5206\u6790\u70b9\n- \u4f7f\u7528\u52a0\u7c97\u6216\u659c\u4f53\u5f3a\u8c03\u91cd\u8981\u4fe1\u606f"""
    )

    process_prompt = PromptTemplate(
        input_variables=["structure"],
        template=
        """\
\u6839\u636e\u5bf9\u96f6\u4ef6\u7684\u7ed3\u6784\u5de5\u827a\u6027\u5206\u6790\u8fdb\u884c\u52a0\u5de5\u5de5\u827a\u8def\u7ebf\u7684\u751f\u6210\uff0c\u8981\u6c42\u8be6\u7ec6\u7684\u52a0\u5de5\u8def\u7ebf\uff0c\u5305\u62ec\uff1a\n1. \u5de5\u5e8f\n2. \u8bbe\u5907\n3. \u52a0\u5de5\u53c2\u6570\n4. \u52a0\u5de5\u65b9\u6cd5\n\n\u8bf7\u4f7f\u7528 Markdown \u683c\u5f0f\u8f93\u51fa\uff0c\u5305\u62ec\uff1a\n- \u4f7f\u7528\u8868\u683c\u5c55\u793a\u5de5\u827a\u8def\u7ebf\n- \u4f7f\u7528\u5217\u8868\u5c55\u793a\u5173\u952e\u53c2\u6570\n- \u4f7f\u7528\u6807\u9898\u7ec4\u7ec7\u4e0d\u540c\u5de5\u5e8f\n- \u4f7f\u7528\u52a0\u7c97\u5f3a\u8c03\u91cd\u8981\u4fe1\u606f"""
    )

    cost_prompt = PromptTemplate(
        input_variables=["process"],
        template=
        """\
\u6839\u636e\u96f6\u4ef6\u7684\u52a0\u5de5\u5de5\u827a\u8def\u7ebf\u751f\u6210\u96f6\u4ef6\u7684\u8be6\u7ec6\u62a5\u4ef7\u8868\uff0c\u5305\u62ec\uff1a\n1. \u5de5\u5e8f\n2. \u8bbe\u5907\n3. \u5de5\u65f6\n4. \u5355\u4f4d\u5de5\u65f6\u4ef7\u683c\n5. \u603b\u4ef7\u683c\n\n\u8bf7\u4f7f\u7528 Markdown \u683c\u5f0f\u8f93\u51fa\uff0c\u5305\u62ec\uff1a\n- \u4f7f\u7528\u8868\u683c\u5c55\u793a\u62a5\u4ef7\u660e\u7ec6\n- \u4f7f\u7528\u5217\u8868\u5c55\u793a\u6210\u672c\u6784\u6210\n- \u4f7f\u7528\u6807\u9898\u7ec4\u7ec7\u4e0d\u540c\u90e8\u5206\n- \u4f7f\u7528\u52a0\u7c97\u5f3a\u8c03\u603b\u4ef7\u548c\u91cd\u8981\u6570\u636e"""
    )

    gcode_prompt = PromptTemplate(
        input_variables=["process"],
        template=
        """\
\u6839\u636e\u52a0\u5de5\u5de5\u827a\u8def\u7ebf\u751f\u6210\u6570\u63a7\u52a0\u5de5G\u4ee3\u7801\uff0c\u8981\u6c42\uff1a\n1. \u6309\u7167\u5de5\u5e8f\u987a\u5e8f\u751f\u6210\u5b8c\u6574\u7684G\u4ee3\u7801\u7a0b\u5e8f\n2. \u5305\u542b\u5200\u5177\u9009\u62e9\u548c\u53c2\u6570\u8bbe\u7f6e\n3. \u5305\u542b\u5750\u6807\u7cfb\u8bbe\u7f6e\u548c\u5de5\u4ef6\u539f\u70b9\n4. \u5305\u542b\u4e3b\u8f6f\u8f6c\u901f\u548c\u8f93\u900f\u901f\u5ea6\n5. \u5fc5\u8981\u7684\u6ce8\u91ca\u8bf4\u660e\n\n\u8bf7\u4f7f\u7528 Markdown \u683c\u5f0f\u8f93\u51fa\uff0c\u5305\u62ec\uff1a\n- \u4f7f\u7528\u4ee3\u7801\u5757\u5c55\u793aG\u4ee3\u7801\n- \u4f7f\u7528\u5217\u8868\u8bf4\u660e\u5173\u952e\u53c2\u6570\n- \u4f7f\u7528\u6807\u9898\u7ec4\u7ec7\u4e0d\u540c\u5de5\u5e8f\u7684\u4ee3\u7801\n- \u4e3a\u6bcf\u4e2a\u91cd\u8981\u4ee3\u7801\u6bb5\u6dfb\u52a0\u6ce8\u91ca\u8bf4\u660e"""
    )

    structure_chain = LLMChain(llm=llm, prompt=structure_prompt, output_key="structure")
    process_chain = LLMChain(llm=llm, prompt=process_prompt, output_key="process")
    cost_chain = LLMChain(llm=llm, prompt=cost_prompt, output_key="cost")
    gcode_chain = LLMChain(llm=llm, prompt=gcode_prompt, output_key="gcode")

    return SequentialChain(
        chains=[structure_chain, process_chain, cost_chain, gcode_chain],
        input_variables=["pdf_data", "json_data"],
        output_variables=["structure", "process", "cost", "gcode"],
    )


def _confirm_with_vlm(result: str, json_data: str, pdf_data: Any) -> Dict[str, Any]:
    """Ask the VLM to validate LLM output against the CAD JSON and PDF."""
    payload = {"result": result, "json": json_data, "pdf": pdf_data}
    return call_vlm(payload)


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze CAD data with a LangChain pipeline and VLM confirmation."""
    json_data = data.get("json")
    if json_data is None:
        raise ValueError("'json' field is required")
    pdf_data = data.get("pdf")

    llm = init_llm()
    chain = _build_chains(llm)

    chain_result: Dict[str, Any] = chain({"pdf_data": pdf_data or "", "json_data": json_data})

    confirmation = _confirm_with_vlm(
        "\n".join(
            [
                chain_result.get("structure", ""),
                chain_result.get("process", ""),
                chain_result.get("cost", ""),
                chain_result.get("gcode", ""),
            ]
        ),
        json_data,
        pdf_data,
    )

    chain_result["confirmation"] = confirmation
    return chain_result

