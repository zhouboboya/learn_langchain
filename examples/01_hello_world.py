"""
═══════════════════════════════════════════════════════════════════
示例 01：Hello World —— 第一次调用大语言模型（LLM）
═══════════════════════════════════════════════════════════════════
目的：用最少的代码，向 LLM 发送一句话，并获得回复。
这是 LangChain 学习的起点。

核心概念：
  - ChatOpenAI：封装了 OpenAI 的聊天模型（也兼容国内代理）
  - invoke()：发送消息并获取回复，这是所有 LangChain 组件的统一入口

运行方式：
  export OPENAI_API_KEY="sk-xxx"       # 设置 API Key
  cd examples && python3 01_hello_world.py
═══════════════════════════════════════════════════════════════════
"""

import os

import os

# 从环境变量加载 API Key（也可手动设置）
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件

# ────────────────────────────────────────
# 第 1 步：创建 LLM 实例（DeepSeek）
# ────────────────────────────────────────
# DeepSeek 兼容 OpenAI 协议，只需修改 base_url 即可
# 模型：deepseek-chat（V3，性价比最高）或 deepseek-reasoner（R1，推理强）
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",   # DeepSeek V3 对话模型
    temperature=0.7,          # 0~1 之间：0=严谨确定，1=天马行空
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
# ────────────────────────────────────────
# 第 2 步：用自然语言提问
# ────────────────────────────────────────
# 就像跟人聊天一样，直接把问题写成字符串即可
resposta = llm.invoke("用一句话解释什么是 LangChain")

# ────────────────────────────────────────
# 第 3 步：查看返回结果
# ────────────────────────────────────────
# resosta 是一个 AIMessage 对象
#   .content   →  纯文本回复
#   .response_metadata  →  token 用量、模型名等元数据
print("【LLM 回复】")
print(resposta.content)
print("\n【元数据（token 用量等）】")
print(resposta.response_metadata)
