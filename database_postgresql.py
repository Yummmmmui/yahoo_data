import pandas as pd
import os
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# 配置
PG_USER = 'postgres'
PG_PASSWORD = '123456'
PG_HOST = '10.0.3.3'
PG_PORT = '5432'
PG_DB_NAME = 'yahoo_data'

# PostgreSQL 连接字符串
PG_CONNECT_STRING = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB_NAME}?client_encoding=utf8"

TABLE_NAME = 'stock_data'  # 数据库表名
MONTHLY_VIEW_NAME = 'stock_monthly_change'  # 月度分析视图名称


def get_db_engine() -> Engine:
    """获取数据库引擎 (使用 SQLAlchemy)"""
    return create_engine(PG_CONNECT_STRING, echo=False)


def create_monthly_change_view():
    """
    创建月度涨跌幅分析视图 stock_monthly_change。
    适配 PostgreSQL 的语法（使用 DATE_TRUNC, TO_CHAR, 和 PostgreSQL 的 LAG 函数）。
    """
    engine = get_db_engine()
    query = f"""
        CREATE OR REPLACE VIEW {MONTHLY_VIEW_NAME} AS
        WITH MonthlyData AS (
            SELECT
                ticker,
                country,
                date::date, -- 确保 Date 列是日期类型
                adj_close AS Monthly_Close, -- 使用小写 adj_close
                -- 提取月份开始日期 (例如: 'YYYY-MM-01')
                DATE_TRUNC('month', date::date) AS Month_Start_Date,
                -- 找到每个月中的最后一天
                ROW_NUMBER() OVER(
                    PARTITION BY ticker, TO_CHAR(date::date, 'YYYY-MM')
                    ORDER BY date DESC
                ) AS rn
            FROM {TABLE_NAME}
        )
        SELECT
            T1.ticker,
            T1.country,
            T1.Month_Start_Date::DATE AS Month_Start_Date,
            T1.Monthly_Close,
            -- 使用 LAG() 获取上个月的 Monthly_Close
            LAG(T1.Monthly_Close, 1) OVER (
                PARTITION BY ticker
                ORDER BY T1.Month_Start_Date
            ) AS Prev_Monthly_Close,
            -- 计算月度涨跌额
            T1.Monthly_Close - LAG(T1.Monthly_Close, 1) OVER (
                PARTITION BY ticker
                ORDER BY T1.Month_Start_Date
            ) AS Monthly_Change_Amt,
            -- 计算月度涨跌幅 (使用 NULLIF 处理分母为 0 或缺失值的情况)
            (T1.Monthly_Close - LAG(T1.Monthly_Close, 1) OVER (
                PARTITION BY ticker
                ORDER BY T1.Month_Start_Date
            )) * 100.0 / NULLIF(LAG(T1.Monthly_Close, 1) OVER (
                PARTITION BY ticker
                ORDER BY T1.Month_Start_Date
            ), 0) AS Monthly_Change_Pct
        FROM MonthlyData T1
        WHERE T1.rn = 1  -- 只保留每个月的最后一条记录 (月底数据)
        ORDER BY T1.ticker, T1.Month_Start_Date;
    """
    try:
        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        print(f"数据库视图 {MONTHLY_VIEW_NAME} 准备就绪。")
    except Exception as e:
        print(f"创建视图 {MONTHLY_VIEW_NAME} 失败: {e}")


def create_table_if_not_exists():
    """
    创建股票数据表 stock_data，如果它不存在的话。
    """
    engine = get_db_engine()

    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            ticker TEXT NOT NULL,
            country TEXT NOT NULL,
            date DATE NOT NULL, 
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            adj_close NUMERIC, 
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        );
    """
    try:
        with engine.connect() as connection:
            connection.execute(text(create_table_query))
            connection.commit()
            print(f"数据库表 {TABLE_NAME} 结构已创建。")
    except Exception as e:
        print(f"创建数据库表 {TABLE_NAME} 失败: {e}")

    create_monthly_change_view()


# ... (load_data_from_db, get_last_date_from_db, execute_and_fetch 保持不变) ...
def load_data_from_db(ticker: str) -> pd.DataFrame:
    """从数据库加载单个股票的完整历史数据。"""
    engine = get_db_engine()
    query = f'SELECT date, open, high, low, close, adj_close, volume FROM {TABLE_NAME} WHERE ticker = :ticker ORDER BY date ASC'
    try:
        df = pd.read_sql_query(text(query), engine, params={'ticker': ticker})
    except Exception as e:
        print(f"加载数据失败: {e}")
        df = pd.DataFrame()
    return df


def get_last_date_from_db(ticker: str) -> str:
    """从数据库获取某个股票的最新日期。"""
    engine = get_db_engine()
    query = f'SELECT date FROM {TABLE_NAME} WHERE ticker = :ticker ORDER BY date DESC LIMIT 1'
    with engine.connect() as connection:
        result = connection.execute(text(query), {'ticker': ticker}).fetchone()
    return result[0] if result else None


def execute_and_fetch(query: str, params: dict = None) -> pd.DataFrame:
    """
    执行 SQL 查询并将结果存储为 Pandas DataFrame。
    """
    engine = get_db_engine()
    try:
        df_result = pd.read_sql_query(text(query), engine, params=params if params else {})
    except Exception as e:
        print(f"执行查询失败")
        return pd.DataFrame()
    return df_result


def get_db_schema() -> str:
    """
    动态获取 stock_data 表和 stock_monthly_change 视图的 schema 信息，
    用于提供给 LLM 进行 SQL 生成。
    """
    engine = get_db_engine()

    # 1. 查询 stock_data 表的结构
    table_query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{TABLE_NAME}' 
        AND table_schema = 'public'
        ORDER BY ordinal_position;
    """
    # 2. 查询 stock_monthly_change 视图的结构
    view_query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{MONTHLY_VIEW_NAME}'
        AND table_schema = 'public'
        ORDER BY ordinal_position;
    """

    schema_info = "PostgreSQL Database Schema:\n"

    try:
        with engine.connect() as connection:
            # 获取表结构
            table_df = pd.read_sql_query(text(table_query), connection)
            schema_info += f"Table: {TABLE_NAME}\n"
            schema_info += "Columns: " + ", ".join([
                f"{row['column_name']} ({row['data_type']})"
                for index, row in table_df.iterrows()
            ]) + "\n\n"

            # 获取视图结构
            view_df = pd.read_sql_query(text(view_query), connection)
            schema_info += f"View: {MONTHLY_VIEW_NAME}\n"
            schema_info += "Columns: " + ", ".join([
                f"{row['column_name']} ({row['data_type']})"
                for index, row in view_df.iterrows()
            ]) + "\n"

    except Exception as e:
        schema_info += f"!!! 警告: 无法获取数据库结构。错误: {e}\n"
        # 如果查询失败，提供一个硬编码的 Fallback Schema (基于 RAG 规则的小写)
        schema_info += "Fallback Schema (硬编码):\n"
        schema_info += f"Table: {TABLE_NAME} (Columns: ticker (text), country (text), date (date), open (numeric), high (numeric), low (numeric), close (numeric), adj_close (numeric), volume (bigint))\n"
        schema_info += f"View: {MONTHLY_VIEW_NAME} (Columns: ticker (text), country (text), month_start_date (date), monthly_close (real), prev_monthly_close (real), monthly_change_amt (real), monthly_change_pct (real))\n"

    return schema_info


def save_data_to_db(df: pd.DataFrame, ticker: str, country: str):
    """
    将 DataFrame 存储到数据库中并处理 UPSERT (ON CONFLICT UPDATE)。
    """
    engine = get_db_engine()

    # 1. 调整列名以匹配数据库
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # 2. 插入股票代码和国家
    df['ticker'] = ticker
    df['country'] = country

    # 确保日期列是 datetime/date 类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 3. 构建临时表名
    # 使用 re.sub 替换所有非字母数字和下划线字符为下划线，确保是一个安全的标识符
    safe_name = re.sub(r'[^a-z0-9_]', '_', f'{ticker}_{country}'.lower())

    # 临时表名必须是小写，且不包含会引起歧义的字符
    temp_table_name_raw = f'temp_{os.getpid()}_{safe_name}'.replace(':', '')

    # 4. 构建 UPSERT SQL 语句的 VALUES 部分
    columns = ', '.join(df.columns)
    update_cols = [col for col in df.columns if col not in ['ticker', 'date', 'country']]
    update_set = ', '.join([f'{col} = excluded.{col}' for col in update_cols])

    if not update_set:
        print(f"Ticker {ticker} 没有可更新的交易数据列。")
        return

    try:
        # 导入到临时表
        df.to_sql(temp_table_name_raw, engine, if_exists='replace', index=False)

        # 从临时表 INSERT/UPDATE 到主表
        upsert_query = f"""
            INSERT INTO {TABLE_NAME} ({columns})
            SELECT {columns} FROM "{temp_table_name_raw}"
            ON CONFLICT (ticker, date) DO UPDATE
            SET {update_set};
        """

        with engine.connect() as connection:
            connection.execute(text(upsert_query))
            connection.commit()

            # 清理临时表
            connection.execute(text(f'DROP TABLE "{temp_table_name_raw}"'))
            connection.commit()

    except Exception as e:
        # 打印详细错误
        print(f"存储 DataFrame 到表 {TABLE_NAME} 失败: {e}")
        # 尝试清理临时表
        try:
            with engine.connect() as connection:
                connection.execute(text(f'DROP TABLE IF EXISTS "{temp_table_name_raw}"'))
                connection.commit()
        except Exception:
            pass


# 初始化数据库结构
if __name__ == '__main__':
    # 确保在运行迁移脚本前，数据库结构已存在

    create_table_if_not_exists()
