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
        input_variables=["json_data"],
        template=
        """
    根据文件中的数据对零件进行结构工艺性分析，需要分析：
    1. 材料的选择和热处理工艺
    2. 尺寸分析精度等级以及加工方法检测方法
    3. 形位公差精度等级以及加工方法检测方法
    4. 表面粗糙度加工方法
    5. 特殊工艺要求

    公差尺寸数据文档：
    {json_data}

    请对每一个尺寸进行详细分析，确保包含全部尺寸。
    请使用 Markdown 格式输出，包括：
    - 使用标题层级（#、##、###）组织内容
    - 使用表格展示数据
    - 使用列表展示分析点
    - 使用加粗或斜体强调重要信息"""
    )

    process_prompt = PromptTemplate(
        input_variables=["structure"],
        template=
        """\
    根据对零件的结构工艺性分析进行加工工艺路线的生成，要求详细的加工路线，包括：
    1. 工序
    2. 设备
    3. 加工参数
    4. 加工方法

    请使用 Markdown 格式输出，包括：
    - 使用表格展示工艺路线
    - 使用列表展示关键参数
    - 使用标题组织不同工序
    - 使用加粗强调重要信息"""
    )


    cost_prompt = PromptTemplate(
        input_variables=["process"],
        template=
        """\
    根据零件的加工工艺路线生成零件的详细报价表，包括：
    1. 工序
    2. 设备
    3. 工时
    4. 单位工时价格
    5. 总价格

    请使用 Markdown 格式输出，包括：
    - 使用表格展示报价明细
    - 使用列表展示成本构成
    - 使用标题组织不同部分
    - 使用加粗强调总价和重要数据"""
    )


    gcode_prompt = PromptTemplate(
        input_variables=["process"],
        template=
        """\
    根据加工工艺路线生成数控加工G代码，要求：
    1. 按照工序顺序生成完整的G代码程序
    2. 包含刀具选择和参数设置
    3. 包含坐标系设置和工件原点
    4. 包含主轴转速和进给速度
    5. 必要的注释说明

    请使用 Markdown 格式输出，包括：
    - 使用代码块展示G代码
    - 使用列表说明关键参数
    - 使用标题组织不同工序的代码
    - 为每个重要代码段添加注释说明"""
    )


    structure_chain = LLMChain(llm=llm, prompt=structure_prompt, output_key="structure")
    process_chain = LLMChain(llm=llm, prompt=process_prompt, output_key="process")
    cost_chain = LLMChain(llm=llm, prompt=cost_prompt, output_key="cost")
    gcode_chain = LLMChain(llm=llm, prompt=gcode_prompt, output_key="gcode")

    return SequentialChain(
        chains=[structure_chain, process_chain, cost_chain, gcode_chain],
        input_variables=["json_data"],
        output_variables=["structure", "process", "cost", "gcode"],
    )


def _confirm_with_vlm(result: str, json_data: str) -> Dict[str, Any]:
    """Ask the VLM to validate LLM output against the CAD JSON."""
    payload = {"result": result, "json": json_data}
    return call_vlm(payload)


def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze CAD data with a LangChain pipeline and VLM confirmation."""

    json_data = data.get("json")
    if json_data is None:
        raise ValueError("'json' field is required")


    llm = init_llm()
    chain = _build_chains(llm)


    chain_result: Dict[str, Any] = chain({"json_data": json_data})

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
    )

    chain_result["confirmation"] = confirmation
    return chain_result


