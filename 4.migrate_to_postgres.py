import os
import pandas as pd
from util.database_postgresql import create_table_if_not_exists, save_data_to_db

MIGRATE_FOLDERS = ['csv', 'new_csv']  # 要迁移的文件夹，后者的优先级更高
COUNTRIES = ['Shanghai_Shenzhen', 'Snp500_Ru1000', 'TSX']


def migrate_csv_to_postgresql_full():
    """遍历指定的 CSV 文件夹并将其内容导入到 PostgreSQL 数据库。
    由于数据库使用了 ON CONFLICT UPDATE (UPSERT)，后处理的数据会覆盖先处理的数据。
    """

    create_table_if_not_exists()

    total_files = 0
    total_rows = 0

    # 按照优先级处理文件夹，确保 'new_csv' 最后处理以覆盖旧数据
    for root_folder in MIGRATE_FOLDERS:
        for country in COUNTRIES:
            country_path = os.path.join(root_folder, country)

            # 检查路径是否存在
            if not os.path.exists(country_path):
                continue

            for file_name in os.listdir(country_path):
                if file_name.endswith('.csv'):
                    # 提取股票代码
                    ticker = file_name.replace('.csv', '')
                    file_path = os.path.join(country_path, file_name)

                    try:
                        # 1. 读取 CSV 文件
                        try:
                            df = pd.read_csv(file_path, parse_dates=['Date'], encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, parse_dates=['Date'], encoding='gbk')
                        # 确保有数据才处理
                        if df.empty:
                            continue
                        rows_count = len(df)
                        total_files += 1

                        # 2. 写入数据库 (save_data_to_db 实现了 PostgreSQL 的 UPSERT 逻辑)
                        save_data_to_db(df, ticker, country)

                        total_rows += rows_count
                        print(f"{root_folder}/{ticker}: 导入 {rows_count} 行数据。")

                    except Exception as e:
                        print(f"导入 {file_path} 失败: {e}")

    print(f"\n总共处理了 {total_files} 个文件，导入/更新了 {total_rows} 行数据。")
    print("数据库已与 CSV 文件同步。")


if __name__ == '__main__':
    # 仅需运行一次，将所有旧数据导入数据库
    migrate_csv_to_postgresql_full()