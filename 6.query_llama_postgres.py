import os
import re
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from util.database_postgresql import execute_and_fetch, get_db_schema

# LLM 和 RAG 配置
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
API_BASE_URL = os.getenv("GROQ_API_BASE", "你的API_BASE_URL")
API_KEY = os.getenv("GROQ_API_KEY", "你的API_KEY")
DASH_SCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的API_KEY")
INDEX_PATH = "llama_index_stock_index"

BASE_FALLBACK_RULES = """
- **视图/表概览:** 数据库使用 **PostgreSQL**，包含 `stock_data` (每日数据) 和 `stock_monthly_change` (月度分析数据) 两个对象。
- **ABSOLUTE CRITICAL RULE (列名引用):** PostgreSQL 列名在 SQL 中必须使用**小写且不加双引号**引用（例如：`ticker`, `month_start_date`, `monthly_change_pct`）。只有包含空格的列名才需要双引号。
- **月度查询规则:** 涉及月度涨跌幅/额必须使用 `stock_monthly_change` 视图。
- **时间序列规则:** 查询最近/最新数据时，必须使用 `ORDER BY [日期字段] DESC`。
"""

# 数据库连接和执行函数 (适配 PostgreSQL)
class DBManager:
    """PostgreSQL 数据库连接管理和执行"""

    def execute_sql_and_fetch(self, query: str) -> pd.DataFrame:
        # 直接调用从 database_postgresql 导入的执行函数
        try:
            return execute_and_fetch(query)
        except Exception as e:
            # 捕获并重新抛出，以便上层逻辑处理
            raise Exception(f"SQL执行失败: {e}")


db_manager = DBManager()

# LangChain 初始化
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=API_BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.0
)


class MockPostgreSQLSchema:
    """提供精确的 PostgreSQL 表和视图 Schema。"""

    def get_table_info(self):
        return (
            "Table stock_data has columns: ticker (TEXT), country (TEXT), date (DATE), open (NUMERIC), high (NUMERIC), low (NUMERIC), close (NUMERIC), adj_close (NUMERIC), volume (BIGINT)."
            "View stock_monthly_change has columns: ticker (TEXT), country (TEXT), month_start_date (DATE), monthly_close (REAL), prev_monthly_close (REAL), monthly_change_amt (REAL), monthly_change_pct (REAL)."
        )


db_schema = MockPostgreSQLSchema()


# RAG 初始化
def initialize_retriever():
    """初始化 RAG 检索器"""
    try:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=DASH_SCOPE_API_KEY
        )
        # Faiss 的加载逻辑保持不变
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Faiss 检索器已加载。RAG 已启用。")
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception:
        print("警告: Faiss 检索器加载失败或索引不存在。请先运行 5.rag_setup.py。")
        return None


retriever = initialize_retriever()

# Text-to-SQL Chain (更新为 PostgreSQL 专家)
SQL_PROMPT_TEMPLATE = """
You are a PostgreSQL expert. Given the table and view schemas and a question, generate the best possible PostgreSQL query.

The tables and views available are: 'stock_data' and 'stock_monthly_change'.
- The view 'stock_monthly_change' contains monthly analysis data, based on your schema: (Ticker, Country, Month_Start_Date, Monthly_Close, Prev_Monthly_Close, Monthly_Change_Amt, Monthly_Change_Pct).
- You MUST use 'stock_monthly_change' for any question involving **monthly change, monthly percentage, or previous month's price**.

The table and view schemas are:
{table_info}

Rules:
1. ONLY output the PostgreSQL query. DO NOT include any explanatory text, markdown quotes (```), or comments.
2. CRITICAL: Column names with spaces MUST be enclosed in double quotes, such as "Adj Close".
3. When filtering by market/region, you MUST use the "Country" column.
4. CRITICAL: To find the single highest (MAX) value of a column (e.g., Close or Monthly_Change_Pct), you MUST select the Ticker and that column, then use 'ORDER BY [Column] DESC LIMIT 1'. Do not use GROUP BY for this task.
{rag_context}
5. **ABSOLUTE CRITICAL OUTPUT RULE:** All queries that return stock data **MUST** include the `Ticker` 和 `Date` (或 `Month_Start_Date`) 字段在 SELECT 子句中。**这是所有结果的强制要求**。

Question: {question}
SQL Query:
"""

SQL_PROMPT = PromptTemplate(
    input_variables=["table_info", "question", "rag_context"],
    template=SQL_PROMPT_TEMPLATE,
)
sql_generation_chain = SQL_PROMPT | llm


# 辅助函数
def clean_sql_output(sql_text: str) -> str:
    """清理 LLM 输出，只保留 SQL 语句"""
    sql_text = re.sub(r'^\s*SQL\s*Query\s*:\s*', '', sql_text, flags=re.IGNORECASE).strip()
    sql_text = re.sub(r'```[sql]*\s*|```', '', sql_text, flags=re.IGNORECASE).strip()
    sql_text = re.sub(r'^\s*(\w+)\s+(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)', r'\2', sql_text,
                      flags=re.IGNORECASE).strip()
    sql_text = re.sub(r'\n', ' ', sql_text).strip()
    sql_text = sql_text.replace('`', '')
    return sql_text


def generate_chart_image(natural_language_query: str, df: pd.DataFrame) -> str:
    """
    使用 Matplotlib 根据查询类型和数据绘制图表，并添加数值标签。
    """
    file_name_prefix = f"chart_{time.strftime('%Y%m%d%H%M%S')}"

    # 尝试从查询中提取股票代码用于图表标题
    ticker_match = re.search(r"'([A-Z0-9]+(?:\.[A-Z]+)?|\w+)'", natural_language_query)
    ticker_for_title = ticker_match.group(1) if ticker_match else "市场/多股"

    if df.empty:
        return "无法绘制图表: 数据结果集为空。"

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False

    # 统一小写列名列表
    df_cols_lower = [col.lower() for col in df.columns]

    is_time_series = any(name in df_cols_lower for name in ['date', 'month_start_date'])
    ticker_col_name = 'ticker' if 'ticker' in df.columns else None
    has_ticker_col = ticker_col_name is not None

    has_multiple_tickers = has_ticker_col and (len(df[ticker_col_name].unique()) > 1)

    # 检查是否为排名请求
    is_ranking_request = "条形图" in natural_language_query or "排名" in natural_language_query or "前" in natural_language_query

    if is_ranking_request and has_ticker_col and 'monthly_change_pct' in df_cols_lower:
        # 强制将结果视为排名数据
        x_col = ticker_col_name
        y_col_candidates = [col for col in df.columns if 'monthly_change_pct' in col.lower()]

        if not y_col_candidates:
            return f"无法绘制图表: 排名数据中找不到 Monthly_Change_Pct 列。"
        y_col = y_col_candidates[0]

        df_plot = df[[x_col, y_col]].copy()
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
        df_plot.dropna(subset=[y_col], inplace=True)

        df_sorted = df_plot.sort_values(by=y_col, ascending=False)

        colors = ['red' if val > 0 else 'green' for val in df_sorted[y_col]]
        rects = ax.bar(df_sorted[x_col], df_sorted[y_col], color=colors)  # 获取 bars 对象

        # 添加数值标签
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:.2f}%'
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    height * 1.01 if height > 0 else height * 0.99,  # 标签位置略微调整
                    label_text,
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, color='black')

        if 'pct' in y_col.lower():
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        ax.tick_params(axis='x', rotation=45)

        ax.set_title(f"市场指标排名 ({ticker_for_title})", fontsize=16)
        ax.set_xlabel(x_col)
        y_label = f'{y_col} (%)' if 'pct' in y_col.lower() else y_col
        ax.set_ylabel(y_label)

        chart_type = "条形图"
        file_path = f"{file_name_prefix}_bar.png"

    elif is_time_series and has_multiple_tickers:
        # 情况 B: 多系列时间序列 (多股对比折线图)
        # 逻辑：查找 Ticker, Date/Month, Value 列
        series_col = ticker_col_name
        potential_date_cols = [col for col in df.columns if col.lower() in ['date', 'month_start_date']]
        y_col_candidate = [col for col in df.columns if 'monthly_change_pct' in col.lower()]
        if not y_col_candidate:
            y_col_candidate = [col for col in df.columns if 'close' in col.lower()]

        if not potential_date_cols or not y_col_candidate:
            return f"无法绘制图表: 无法从 {df.columns} 中识别 Ticker, Date/Month 和 Value 列。"

        x_col = potential_date_cols[0]
        y_col = y_col_candidate[0]

        df_plot = df[[series_col, x_col, y_col]].copy()
        try:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
        except ValueError:
            return f"无法绘制图表: 日期列 '{x_col}' 转换失败，请检查数据格式。"

        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
        df_plot.dropna(subset=[y_col], inplace=True)

        # 添加数值标签
        for name, group in df_plot.groupby(series_col):
            ax.plot(group[x_col], group[y_col], marker='o', linestyle='-', label=name, linewidth=2)
            for x, y in zip(group[x_col], group[y_col]):
                label_text = f'{y:.2f}%'
                color = 'red' if y > 0 else 'green'
                ax.text(x, y + 0.5 if y >= 0 else y - 1.5, label_text,
                        ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=10, color=color, alpha=0.9)

        if 'pct' in y_col.lower():
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        ax.set_title(f"股票指标对比 ({ticker_for_title})", fontsize=16)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f'{y_col}')
        ax.legend(title=series_col)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(rotation=45)

        chart_type = "多股/多指标对比折线图"
        file_path = f"{file_name_prefix}_multi_line.png"

    elif is_time_series and len(df.columns) >= 2:
        # 情况 A: 标准单系列时间序列

        date_cols = [col for col in df.columns if col.lower() in ['date', 'month_start_date']]
        value_cols = [col for col in df.columns if col not in date_cols and col.lower() != 'ticker']

        # 简化列名选择逻辑，防止因列名数量变化而失败
        if len(date_cols) >= 1 and len(value_cols) >= 1:
            x_col = date_cols[0]
            y_col = value_cols[0]
        else:
            # 最后的检查和回退逻辑
            potential_x = [col for col in df.columns if col.lower() in ['date', 'month_start_date']]
            potential_y = [col for col in df.columns if col not in potential_x and col.lower() != 'ticker']

            if len(potential_x) >= 1 and len(potential_y) >= 1:
                x_col = potential_x[0]
                y_col = potential_y[0]
            else:
                return f"无法绘制图表: 无法从 {df.columns} 中识别日期和值列。"

        df_plot = df[[x_col, y_col]].copy()

        try:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
        except ValueError:
            return f"无法绘制图表: 日期列 '{x_col}' 转换失败，请检查数据格式。"

        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
        df_plot.dropna(subset=[y_col], inplace=True)

        ax.plot(df_plot[x_col], df_plot[y_col], marker='o', linestyle='-', color='#1f77b4', linewidth=2)

        # 添加数值标签
        if 'pct' in y_col.lower():
            for x, y in zip(df_plot[x_col], df_plot[y_col]):
                label_text = f'{y:.2f}%'
                color = 'red' if y > 0 else 'green'
                ax.text(x, y + 0.5 if y >= 0 else y - 1.5, label_text,
                        ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=12, color=color)

        if 'pct' in y_col.lower():
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        ax.set_title(f"股票指标 ({ticker_for_title})", fontsize=16)
        ax.set_xlabel(x_col)
        y_label = f'{y_col} (%)' if 'pct' in y_col.lower() else y_col
        ax.set_ylabel(y_label)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(rotation=45)

        chart_type = "折线图"
        file_path = f"{file_name_prefix}_line.png"

    # 处理未被前面条件捕获的简单排名数据 (非时间序列)
    elif len(df.columns) >= 2 and not is_time_series:
        # 情况 C: 排名/比较数据 (条形图)
        x_col = df.columns[0]
        y_col = df.columns[1]

        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df.dropna(subset=[y_col], inplace=True)

        df_sorted = df.sort_values(by=y_col, ascending=False)

        colors = ['red' if val > 0 else 'green' for val in df_sorted[y_col]]
        rects = ax.bar(df_sorted[x_col], df_sorted[y_col], color=colors)

        # 添加数值标签 (条形图)
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:.2f}%' if 'pct' in y_col.lower() else f'{height:.2f}'
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    height * 1.01 if height > 0 else height * 0.99,
                    label_text,
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, color='black')

        if 'pct' in y_col.lower():
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

        ax.set_title("市场指标排名", fontsize=16)
        ax.set_xlabel(x_col)
        y_label = f'{y_col} (%)' if 'pct' in y_col.lower() else y_col
        ax.set_ylabel(y_label)

        ax.tick_params(axis='x', rotation=45)

        chart_type = "条形图"
        file_path = f"{file_name_prefix}_bar.png"

    else:
        return f"无法绘制图表: 数据格式不符合预期 (列数: {len(df.columns)})。"

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    os.makedirs('chart', exist_ok=True)
    full_path = os.path.join('chart', file_path)
    plt.savefig(full_path, dpi=300)
    plt.close(fig)

    return f"图表已生成: {full_path}"


# 核心查询函数
def query_stock_data_with_llm(natural_language_query: str):
    start_query_time = time.time()
    rag_context = ""  # 初始化 rag_context 变量

    # 1. 初始化 LLM 和 RAG 设置
    try:
        # LLM 初始化 (用于 SQL 生成)
        llm = ChatOpenAI(
            openai_api_base=API_BASE_URL,
            openai_api_key=API_KEY,
            model=LLM_MODEL_NAME,
            temperature=0
        )

        # LlamaIndex 嵌入模型和 RAG 设置
        embedding_model = DashScopeEmbedding(
            api_key=DASH_SCOPE_API_KEY,
            model_name="text-embedding-v2"
        )
        Settings.embed_model = embedding_model

        print(f"\n原始查询: {natural_language_query}")

        # 2. RAG 检索上下文 (使用 LlamaIndex)
        if not os.path.exists(INDEX_PATH):
            return "查询失败! LlamaIndex 索引未找到。请先运行 5.rag_setup.py 文件。"

        # 加载 LlamaIndex 索引
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context=storage_context)

        # 使用 LlamaIndex Retriever 获取 RAG 上下文
        retriever = index.as_retriever(similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(natural_language_query)

        # 格式化检索结果
        context_list = [f"- RAG Rule: {node.text.replace('**', '')}" for node in retrieved_nodes]
        rag_context = "\n\nRAG 检索到的上下文:\n" + "\n".join(context_list)
        print(rag_context)

        # 3. LLM生成 SQL 语句

        # 构建 LCEL 链
        sql_generation_chain = SQL_PROMPT | llm

        # 准备输入字典
        chain_input = {
            "table_info": get_db_schema(),
            "question": natural_language_query,
            "rag_context": BASE_FALLBACK_RULES + rag_context  # 组合基础规则和检索到的规则
        }

        # 调用 LCEL 链
        llm_response_content = sql_generation_chain.invoke(chain_input)

        # 提取内容
        if hasattr(llm_response_content, 'content'):
            sql_output_text = llm_response_content.content
        else:
            sql_output_text = str(llm_response_content)

        # 清理 SQL 语句
        sql_query = clean_sql_output(sql_output_text)
        print(f"\n生成的 SQL:\n{sql_query}")

        # 4. 执行 SQL
        print("正在执行 SQL...")
        # 使用 PostgreSQL 数据库执行函数
        result_df = execute_and_fetch(sql_query)

        if result_df.empty:
            result_output = "查询结果为空。"
        else:
            result_output = result_df.to_string(index=False)

        # 5. 生成图表
        if "请生成" in natural_language_query or "画出" in natural_language_query or "图表" in natural_language_query:
            image_output = generate_chart_image(natural_language_query, result_df)
        else:
            image_output = "无需生成图表。"

        # 6. 整合最终输出
        final_output = (
            f"\n--- 结果 (总耗时 {time.time() - start_query_time:.2f}秒) ---\n"
            f"{result_output}\n"
            f"{image_output}\n"
        )
        return final_output

    except Exception as e:
        return f"查询失败! 错误信息: {e}"


if __name__ == "__main__":
    print(f"使用的 LLM 模型: {LLM_MODEL_NAME} (Groq)")
    print(f"使用的数据库: PostgreSQL")

    print("\n" + "=" * 50 + "\n")

    query1 = "查询 '000001.SZ' 最近 6 个月的月度涨跌幅百分比 (Monthly_Change_Pct) 和日期 (Month_Start_Date)。请生成一张清晰的折线图。"
    result1 = query_stock_data_with_llm(query1)
    print(result1)

    query2 = "在 'Shanghai_Shenzhen' 市场中，2025年10月月度涨幅百分比 (Monthly_Change_Pct) 前十？请生成一张条形图。"
    result2 = query_stock_data_with_llm(query2)
    print(result2)

    query3 = "查询 'BABA' 股票月度涨幅百分比 (Monthly_Change_Pct) 和日期 (Month_Start_Date) 的最近12条记录。请生成一张折线图。"
    result3 = query_stock_data_with_llm(query3)
    print(result3)

    query4 = "关联stock_data和stock_monthly_change，查询 'BABA' 2025 年的每月收盘价（Close）及该年的月度涨跌幅（Monthly_Change_Pct）"
    result4 = query_stock_data_with_llm(query4)
    print(result4)

    query5 = "查询 'AAP' 和 'AAPL' 最近 3 个月的月度涨跌幅（Monthly_Change_Pct），按日期和股票代码分组。"
    result5 = query_stock_data_with_llm(query5)
    print(result5)