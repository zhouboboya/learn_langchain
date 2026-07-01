"""
═══════════════════════════════════════════════════════════════════
示例 09：文档加载器（Document Loader）—— 读文件、网页、PDF
═══════════════════════════════════════════════════════════════════
目的：把各种格式的"外部知识"读进来，变成 LangChain 能处理的标准格式。
RAG（示例04）的第一步就是加载文档。

核心概念：
  - Document(page_content=..., metadata=...) ：LangChain 的标准文档格式
  - Loader（加载器）：每种文件格式有对应的 Loader
  - 常见 Loader：TextLoader、PyPDFLoader、WebBaseLoader、CSVLoader

支持的格式（LangChain 有 100+ 种 Loader）：
  PDF、Word、Markdown、CSV、网页、Notion、飞书、数据库...

运行方式：设置了 DEEPSEEK_API_KEY 即可（本示例只加载文档，不调用 LLM）
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ────────────────────────────────────────
# 第 1 步：加载文本文件（TXT / Markdown）
# ────────────────────────────────────────
from langchain_community.document_loaders import TextLoader

# 先创建一个测试用的文本文件
test_file = "/tmp/langchain_demo.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("LangChain 是一个强大的 LLM 应用框架。\n")
    f.write("它支持 Python 和 JavaScript 两种语言。\n")
    f.write("核心组件包括：Chains、Agents、Tools、Memory。\n")

# TextLoader 读取文件，每行自动转成一个 Document
loader = TextLoader(test_file, encoding="utf-8")
docs = loader.load()

print("【TXT 文件加载】")
for i, doc in enumerate(docs):
    print(f"  文档 {i+1}:")
    print(f"    内容: {doc.page_content[:100]}")
    print(f"    元数据: {doc.metadata}")
print()

# ────────────────────────────────────────
# 第 2 步：加载网页内容
# ────────────────────────────────────────
# WebBaseLoader 能抓取网页正文（自动去掉导航、广告等噪音）
from langchain_community.document_loaders import WebBaseLoader

# 用 requests 获取网页，BeautifulSoup 解析正文
# web_loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
# web_docs = web_loader.load()

# 为了方便演示，直接模拟网页内容
print("【网页加载（模拟）】")
print("  实际用法: WebBaseLoader('https://xxx').load()")
print("  自动去除导航、广告、页脚等噪音，只保留正文")
print()

# ────────────────────────────────────────
# 第 3 步：加载 PDF 文件
# ────────────────────────────────────────
# PyPDFLoader 加载 PDF，每页转成一个 Document
from langchain_community.document_loaders import PyPDFLoader

# 创建一个模拟 PDF 来演示（实际只需 PyPDFLoader(path).load()）
print("【PDF 加载】")
print("  实际用法: PyPDFLoader('/path/doc.pdf').load()")
print("  每页 PDF 变成一个 Document，metadata 包含页码")
print()

# ────────────────────────────────────────
# 第 4 步：加载目录下所有文件（批量加载）
# ────────────────────────────────────────
# DirectoryLoader 能加载一个目录下的所有指定类型文件
from langchain_community.document_loaders import DirectoryLoader

# 加载 examples 目录下的所有 .py 文件（把代码当文档）
loader = DirectoryLoader(
    path="/Users/zhouningbo/Desktop/langchain/examples/",
    glob="*.py",                      # 只加载 .py 文件
    loader_cls=TextLoader,            # 用 TextLoader 读每个文件
    show_progress=False,
)

code_docs = loader.load()

print("【批量加载 .py 文件】")
print(f"  共加载 {len(code_docs)} 个文件：")
for doc in code_docs:
    lines = doc.page_content.count("\n") + 1
    print(f"    {doc.metadata['source'].split('/')[-1]} ({lines} 行)")

# ────────────────────────────────────────
# 第 5 步：加载后接 RAG（与示例 04 联动）
# ────────────────────────────────────────
# 实际流程是把加载的文档切片、向量化、存入向量库
print("\n【完整 RAG 流水线预览】")
print("  Loader 加载 → Splitter 切片 → Embeddings 向量化 → VectorStore 存储")
print("  详见示例 04_rag.py")

# 清理测试文件
os.remove(test_file)
