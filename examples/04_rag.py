"""
═══════════════════════════════════════════════════════════════════
示例 04：RAG（检索增强生成）
═══════════════════════════════════════════════════════════════════
目的：让 LLM 能回答"它没学过"的知识（你自己的文档、PDF 等）。

RAG 是 LangChain 最实用的场景，核心流程四步走：

  文档加载  →  文本切分  →  向量存储  →  检索 + 生成
  (Loader)     (Splitter)   (VectorDB)    (Retrieve → LLM)

  1. 文档加载：读取 PDF / TXT / 网页 / 数据库
  2. 文本切分：把长文档切成小段，每段几百字
  3. 向量化存储：每段文本转成"向量"存入向量数据库
  4. 提问时：先检索相关内容 → 作为上下文喂给 LLM → 生成答案

运行方式：同 01，设置 OPENAI_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ────────────────────────────────────────
# 注意：DeepSeek 没有 Embedding API
# ────────────────────────────────────────
# 文本向量化使用免费的本地模型（sentence-transformers）
# 如果你有 OpenAI Key，也可以改用 OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ────────────────────────────────────────
# 第 1 步：准备"知识库"（模拟文档）
# ────────────────────────────────────────
# 实际场景中你会用 DocumentLoader 读文件：
#   from langchain_community.document_loaders import TextLoader, PyPDFLoader
# 这里用模拟数据演示原理

documents = [
    "LangChain 是一个用于构建 LLM 应用的开源框架，支持 Python 和 JS。",
    "LangChain 的核心概念包括：Chain（链）、Agent（智能体）、Tool（工具）、Memory（记忆）。",
    "RAG 是 Retrieval-Augmented Generation 的缩写，意思是检索增强生成。",
    "LangSmith 是 LangChain 旗下的 LLM 应用调试与监控平台，可以追踪每次调用。",
    "LCEL 全称 LangChain Expression Language，用管道符 | 串联组件，是 LangChain 最推荐的写法。",
    "向量数据库是一种专门存储和检索向量的数据库，常用有 Chroma、FAISS、Pinecone、Weaviate。",
]

# ────────────────────────────────────────
# 第 2 步：文本切分
# ────────────────────────────────────────
# TextSplitter 按字符数切分，overlap 表示相邻段之间的重复字数
# 有重叠能让"跨段"的信息不被打断
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每段最多 200 个字符
    chunk_overlap=30,     # 相邻段重叠 30 个字符
    separators=["\n", "。", "，", " ", ""],  # 优先按这些分隔符切
)

chunks = text_splitter.create_documents(documents)
print(f"【切片完成】共 {len(chunks)} 个片段\n")

# ────────────────────────────────────────
# 第 3 步：向量化 + 存入向量数据库
# ────────────────────────────────────────
# OpenAIEmbeddings 把"文字"转成"向量"（一串数字）
# Chroma 是轻量级向量数据库，数据存在内存中，学习最方便
from langchain_chroma import Chroma

# 创建嵌入模型（本地免费，无需 API Key）
# 首次运行会下载模型文件（约 100MB），之后使用本地缓存
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 把所有文档片段向量化后存入 Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_learning",
)

# .as_retriever() 把向量库包装成"检索器"
# k=3 表示取出最相似的 3 条结果
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("【检索测试】")
test_results = retriever.invoke("什么是 LCEL？")
for i, doc in enumerate(test_results):
    print(f"  结果 {i+1}: {doc.page_content[:80]}...")
print()

# ────────────────────────────────────────
# 第 4 步：构建 RAG 链
# ────────────────────────────────────────
# 提示词中包含 {context} 和 {question} 两个变量
rag_prompt = ChatPromptTemplate.from_template("""\
根据以下资料回答问题。如果资料中没有相关信息，请如实说明不知道。

资料：
{context}

问题：{question}

回答：""")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 构建 RAG 链
# RunnablePassthrough() 意思是"原样透传"——把用户问题原封不动传给 {question}
rag_chain = (
    {
        "context": retriever,                     # ← 检索到的资料填入 {context}
        "question": RunnablePassthrough(),         # ← 用户问题原样填入 {question}
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────
# 第 5 步：提问！
# ────────────────────────────────────────
print("【RAG 问答】")

questions = [
    "什么是 RAG？",
    "LangChain 的核心概念有哪些？",
]

for q in questions:
    answer = rag_chain.invoke(q)
    print(f"\n问：{q}")
    print(f"答：{answer}")
