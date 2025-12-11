import os
import time
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.dashscope import DashScopeEmbedding
from typing import List

# 配置
DASH_SCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的API KEY")
EMBEDDING_MODEL_NAME = "text-embedding-v2"
INDEX_PATH = "llama_index_stock_index"

BASE_FALLBACK_RULES = """
- **视图/表概览:** 数据库使用 **PostgreSQL**，包含 `stock_data` (每日数据) 和 `stock_monthly_change` (月度分析数据) 两个对象。
- **ABSOLUTE CRITICAL RULE (列名引用):** PostgreSQL 列名在 SQL 中必须使用**小写且不加双引号**引用（例如：`ticker`, `month_start_date`, `monthly_change_pct`）。只有包含空格的列名才需要双引号。
- **月度查询规则:** 涉及月度涨跌幅/额必须使用 `stock_monthly_change` 视图。
- **时间序列规则:** 查询最近/最新数据时，必须使用 `ORDER BY [日期字段] DESC`。
"""


# 1. 知识源 (Documents)
def define_rag_documents() -> List[Document]:
    """定义细粒度 RAG 知识文档列表，使用 LlamaIndex Document。"""
    documents = [
        # 数据库概览及字段
        Document(text="数据库包含两个主要对象：`stock_data` 表和 `stock_monthly_change` 视图。**底层数据库使用 PostgreSQL**。"),
        Document(text="`stock_data` 表包含每日股票数据，字段有：ticker (TEXT, PK1), date (DATE, PK2), country (TEXT), open (NUMERIC), high (NUMERIC), low (NUMERIC), close (NUMERIC), adj_close (NUMERIC), volume (BIGINT)。"),
        Document(text="`stock_monthly_change` 视图包含月度分析数据，字段有：ticker (TEXT, PK1), country (TEXT), month_start_date (TIMESTAMP), monthly_close (REAL), prev_monthly_close (REAL), monthly_change_amt (REAL), monthly_change_pct (REAL)。"),

        Document(text="**ABSOLUTE CRITICAL POSTGRES RULE (列名引用):** PostgreSQL 列名在 SQL 中必须使用**小写**且**不加双引号**引用。例如，必须使用 `ticker`, `month_start_date`, `monthly_change_pct`。**核心原则：除非包含空格，否则一律小写且不加引号。**"),

        # 强制输出规则
        Document(text="**ABSOLUTE CRITICAL OUTPUT RULE (结果字段):** 所有查询结果（无论是否涉及图表）**必须**在 SELECT 子句中包含 `ticker` 字段和相应的日期字段 (`date` 或 `month_start_date`)，以确保结果清晰和图表生成成功。"),
        Document(text="**ABSOLUTE CRITICAL OUTPUT FORMAT:** LLM 必须且只能输出最终的 PostgreSQL SQL 语句。**严禁**在 SQL 语句块内或周围包含任何注释、解释、Markdown 格式化外的文字（如 'However...', 'To achieve that...' 等）。"),

        # 关键联接规则
        Document(text="**ABSOLUTE CRITICAL POSTGRES RULE (联接稳健性):** 联接 `stock_data` (T1) 和 `stock_monthly_change` (T2) 时，联接条件**必须**使用双向截断：`T1.ticker = T2.ticker AND DATE_TRUNC('month', T1.date) = DATE_TRUNC('month', T2.month_start_date)`。"),

        # 关键日期过滤规则
        Document(text="**CRITICAL POSTGRES RULE (TIMESTAMP 筛选):** `stock_monthly_change.month_start_date` 字段是 **TIMESTAMP** 类型。查询或比较该字段时，日期字面量（如 '2024-10-01'）**必须**使用 `::timestamp` 后缀进行明确转换，例如：`month_start_date = 'YYYY-MM-DD'::timestamp`。"),
        Document(text="**CRITICAL POSTGRESQL DATE RULE 1 (过滤年份):** PostgreSQL 过滤年份的正确语法是：`EXTRACT(YEAR FROM [日期字段]) = [年份]`。"),
        Document(text="**CRITICAL POSTGRESQL DATE RULE 2 (过滤月份):** PostgreSQL 过滤月份的正确语法是：`EXTRACT(MONTH FROM [日期字段]) = [月份]`。"),

        # 联接粒度对齐规则
        Document(text="**ABSOLUTE CRITICAL RULE (联接粒度对齐):** 当联接 `stock_data` (每日) 和 `stock_monthly_change` (月度) 并且需要精确到月度收盘价时，**必须**同时满足三个条件：1) Ticker 匹配；2) 双向截断联接：`DATE_TRUNC('month', T1.date) = DATE_TRUNC('month', T2.month_start_date)`；3) 仅取月末日期过滤：`T1.date = (SELECT MAX(date) FROM stock_data sd_sub WHERE sd_sub.ticker = T1.ticker AND DATE_TRUNC('month', sd_sub.date) = DATE_TRUNC('month', T1.date))`。"),

        # 月度视图的使用限制
        Document(text="**视图使用限制:** `stock_monthly_change` 视图已经是月度聚合数据，查询某股票某个月的每日收盘价时，**必须**使用 `stock_data` 表。"),

        # Top N 规则
        Document(text="**Top N 规则:** 查询某列的最大值（如最大涨幅）或前 N 记录时，**必须**使用 `ORDER BY [列名] DESC LIMIT N`，且 SELECT 字段中必须包含 ticker 和日期。"),

        # 联接示例模板
        Document(text="**CRITICAL EXAMPLE (复杂联接):** 查询 'PARA' 2025年的每月收盘价 (`close`) 及其月度涨幅。唯一正确的 SQL 模板是：`SELECT T1.ticker, T1.date, T1.close, T2.monthly_change_pct FROM stock_data AS T1 JOIN stock_monthly_change AS T2 ON T1.ticker = T2.ticker AND DATE_TRUNC('month', T1.date) = DATE_TRUNC('month', T2.month_start_date) WHERE T1.ticker = 'PARA' AND EXTRACT(YEAR FROM T1.date) = 2025 AND T1.date = (SELECT MAX(date) FROM stock_data sd_sub WHERE sd_sub.ticker = T1.ticker AND DATE_TRUNC('month', sd_sub.date) = DATE_TRUNC('month', T1.date)) ORDER BY T1.date`。"),
    ]
    return documents


def setup_rag_index_llamaindex():
    """
    设置并持久化 LlamaIndex 索引。
    """
    new_documents = define_rag_documents()

    # 1. 初始化 LlamaIndex 嵌入模型
    try:
        embedding_model = DashScopeEmbedding(
            api_key=DASH_SCOPE_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        Settings.embed_model = embedding_model
    except Exception as e:
        print(f"嵌入模型初始化失败，请检查配置和API Key: {e}")
        return

    # 2. 检查索引是否已存在，如果存在则直接加载或跳过创建
    index_storage_path = os.path.join(INDEX_PATH, 'docstore.json')

    if os.path.exists(index_storage_path):
        print(f"检测到现有索引 {INDEX_PATH}，正在加载...")
        try:
            # 使用 load_index_from_storage 加载索引
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
            index = load_index_from_storage(storage_context=storage_context)
            print(f"索引已成功加载。")
            return  # 加载成功则退出，不进行后续的创建和持久化

        except Exception as e:
            print(f"加载现有索引失败：{e} (将尝试新建索引)")

    # 3. 创建 LlamaIndex 索引
    print(f"未检测到现有索引或加载失败，正在创建新的 LlamaIndex 索引，包含 {len(new_documents)} 条文档...")

    index = VectorStoreIndex.from_documents(new_documents)

    # 4. 持久化 LlamaIndex 索引
    print(f"索引创建完成，正在持久化到 {INDEX_PATH}...")
    index.storage_context.persist(persist_dir=INDEX_PATH)


if __name__ == "__main__":
    start_time = time.time()
    setup_rag_index_llamaindex()
    end_time = time.time()
    print(f"\nRAG索引设置完成，总耗时: {end_time - start_time:.2f} 秒")