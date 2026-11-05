"""
Domain-adapted prompts for each dataset.
Base prompts from original SQL scripts, adapted to:
- ds_1000 (code): Python pandas code generation
- ddxplus (text): medical diagnosis
- toolbench (json): API selection
"""

# ===================== Zero-shot prompts =====================
ZERO_SHOT_PROMPTS = {
    "ds_1000": """你是Python编程专家。根据用户问题生成正确的代码。
用户问题：{question}
输出要求：只输出代码，不要解释。""",

    "ddxplus": """你是医学诊断专家。根据患者信息给出最可能的诊断结果。
患者信息：{question}
输出要求：只输出诊断结果名称，不要解释。""",

    "toolbench": """你是工具调用专家。根据用户问题选择最合适的API并输出调用参数。
用户问题：{question}
输出要求：输出API名称和JSON格式的调用参数，不要解释。""",
}


# ===================== Few-shot examples =====================
FEW_SHOT_EXAMPLES = {
    "ds_1000": [
        {"question": "Select rows where Col1 > 5 from the DataFrame",
         "answer": "df[df['Col1'] > 5]"},
        {"question": "Sort the DataFrame by Col2 in descending order",
         "answer": "df.sort_values('Col2', ascending=False)"},
        {"question": "Calculate the mean of Col3 grouped by Type",
         "answer": "df.groupby('Type')['Col3'].mean()"},
    ],
    "ddxplus": [
        {"question": "患者男性，65岁，突发胸痛伴大汗，有高血压病史",
         "answer": "急性心肌梗死"},
        {"question": "患者女性，30岁，转移性右下腹痛伴发热，麦氏点压痛",
         "answer": "急性阑尾炎"},
        {"question": "患者儿童，发热3天，口腔黏膜斑，皮肤红色斑丘疹",
         "answer": "麻疹"},
    ],
    "toolbench": [
        {"question": "What is the weather like in Beijing today?",
         "answer": "API_name: get_weather\n参数: {\"city\": \"Beijing\"}"},
        {"question": "Calculate the square root of 144",
         "answer": "API_name: calculate_sqrt\n参数: {\"number\": 144}"},
        {"question": "Send an email to john@example.com with subject 'Hello'",
         "answer": "API_name: send_email\n参数: {\"to\": \"john@example.com\", \"subject\": \"Hello\"}"},
    ],
}


def build_few_shot_prompt(dataset_name: str, question: str, num_shots: int = 3) -> str:
    examples = FEW_SHOT_EXAMPLES[dataset_name][:num_shots]
    example_text = ""
    for ex in examples:
        example_text += f"用户问题：{ex['question']}\n{ex['answer']}\n\n"

    if dataset_name == "ds_1000":
        return f"""以下是几个Python代码生成示例：
{example_text}
请根据用户问题生成正确的Python代码，只输出代码，不要解释。
用户问题：{question}
代码："""
    elif dataset_name == "ddxplus":
        return f"""以下是几个医学诊断示例：
{example_text}
请根据患者信息给出最可能的诊断结果，只输出诊断名称，不要解释。
患者信息：{question}
诊断结果："""
    else:  # toolbench
        return f"""以下是几个工具调用示例：
{example_text}
请根据用户问题选择合适的API并输出调用参数，不要解释。
用户问题：{question}
API调用："""


# ===================== CoT prompts =====================
COT_PROMPTS = {
    "ds_1000": """请一步步思考如何回答用户的问题，最后输出对应的 Python 代码（用```python...```包裹）。
用户问题：{question}
逐步推理：""",

    "ddxplus": """请一步步思考患者的症状和可能的病因，最后给出最可能的诊断结果。
患者信息：{question}
逐步推理：""",

    "toolbench": """请一步步思考用户的需求，分析需要调用什么工具，最后输出API名称和调用参数。
用户问题：{question}
逐步推理：""",
}


# ===================== GrowPrompt prompt builder =====================
def build_grow_prompt(dataset_name: str, question: str, window_examples: list) -> str:
    """
    window_examples: list of (question, pred_answer, is_correct) tuples
    """
    example_text = ""
    for i, (ex_q, ex_ans, correct) in enumerate(window_examples):
        feedback = "回答正确" if correct else "回答不正确"
        example_text += f"历史问题{i+1}：{ex_q}\n模型输出：{ex_ans}\n反馈：{feedback}\n"

    if dataset_name == "ds_1000":
        task = "Python代码"
    elif dataset_name == "ddxplus":
        task = "诊断结果"
    else:
        task = "API调用"

    return f"""以下是历史问答记录及正确性反馈，请参考后回答新问题。
{example_text}
现在处理新问题，只输出{task}，不要解释。
用户问题：{question}
{task}："""


# ===================== Memory context builders =====================
def build_memory_prompt(dataset_name: str, question: str, memory: list) -> str:
    """
    Build prompt with memory examples.
    memory: list of {"q": str, "a": str}
    """
    memory_text = ""
    for m in memory:
        memory_text += f"问题：{m['q']}\n答案：{m['a']}\n"

    if dataset_name == "ds_1000":
        task = "Python代码"
    elif dataset_name == "ddxplus":
        task = "诊断结果"
    else:
        task = "API调用"

    return f"""你是专业的AI助手。参考以下示例回答新问题。
参考示例：
{memory_text if memory_text else "无"}
用户问题：{question}
只输出{task}，不要解释。"""


# ===================== ASK-StreamDiag subtask definitions =====================
SUBTASKS_BY_DATASET = {
    "ds_1000": [
        {"name": "problem_understanding", "desc": "正确理解用户的编程需求和目标"},
        {"name": "dataframe_structure", "desc": "正确识别DataFrame的结构、列名和数据类型"},
        {"name": "pandas_operations", "desc": "正确选择和实现Pandas操作（筛选、分组、聚合等）"},
        {"name": "code_correctness", "desc": "生成的代码语法正确且能正确运行"},
    ],
    "ddxplus": [
        {"name": "symptom_identification", "desc": "正确识别患者的关键症状和体征"},
        {"name": "differential_diagnosis", "desc": "正确列出可能的鉴别诊断"},
        {"name": "diagnostic_reasoning", "desc": "基于症状和病史的诊断推理逻辑正确"},
        {"name": "final_diagnosis", "desc": "最终诊断结果与金标一致"},
    ],
    "toolbench": [
        {"name": "intent_recognition", "desc": "正确理解用户的意图和需求"},
        {"name": "api_selection", "desc": "正确选择最合适的API工具"},
        {"name": "parameter_extraction", "desc": "正确提取API调用所需的参数"},
        {"name": "argument_format", "desc": "参数格式和类型正确"},
    ],
}


# ===================== Judge prompts =====================
def build_judge_prompt(dataset_name: str, question: str, generated_answer: str,
                       subtask_desc: str) -> str:
    if dataset_name == "ds_1000":
        domain = "Python代码"
    elif dataset_name == "ddxplus":
        domain = "医学诊断"
    else:
        domain = "工具调用"

    return f"""你是一个严格的{domain}评估专家。请判断以下生成的答案是否满足指定的子任务要求。

用户问题：{question}
生成的答案：{generated_answer}
子任务描述：{subtask_desc}

请只输出 JSON 格式，不要有其他内容：
{{"success": true/false, "reason": "简短原因"}}
"""


# ===================== Prompt perturbation =====================
def build_rephrase_prompt(question: str) -> str:
    return f"""请将以下问题用不同的措辞重新表述一遍，保持语义完全相同，只输出改写后的问题：
原问题：{question}
改写后的问题："""


def build_add_example_suffix(dataset_name: str) -> str:
    if dataset_name == "ds_1000":
        return "\n例如：df[df['col'] > 0] 是筛选操作。"
    elif dataset_name == "ddxplus":
        return "\n例如：发热+咳嗽 → 上呼吸道感染。"
    else:
        return "\n例如：查询天气 → get_weather(city='Beijing')。"


# ===================== Dispatcher (subtask decomposition) prompt =====================
def build_decomposition_prompt(dataset_name: str, question: str) -> str:
    if dataset_name == "ds_1000":
        return f"""任务：将用户的编程问题拆解为代码生成所需的子步骤：
用户问题：{question}
请输出：
1. 需要理解的问题目标
2. 涉及的DataFrame结构
3. 需要执行的Pandas操作
4. 输出格式要求"""
    elif dataset_name == "ddxplus":
        return f"""任务：将患者的诊断问题拆解为诊断推理所需的子步骤：
患者信息：{question}
请输出：
1. 关键症状提取
2. 可能的鉴别诊断
3. 诊断推理逻辑
4. 最终诊断"""
    else:
        return f"""任务：将用户的工具调用问题拆解为API调用所需的子步骤：
用户问题：{question}
请输出：
1. 用户意图识别
2. 需要调用的API
3. 需要提取的参数
4. 参数格式要求"""
