import os
import time
import datetime
import sqlite3
import pandas as pd
import requests
import yfinance as yf
from calendar import monthrange

print(f"yfinance 版本: {yf.__version__}")

# 定义全局变量 fail_download
fail_download = {'Snp500_Ru1000': []}

def downloader(ticker, start_date, end_date):
    """
    下载逻辑：只管下载并保存为 CSV
    """
    try:
        data = yf.Ticker(ticker).history(period="max", interval="1d", start=start_date, end=end_date)
        if data is not None and len(data) > 1:
            os.makedirs('new_csv/Snp500_Ru1000', exist_ok=True)
            # 存成 CSV 文件
            filepath = f"new_csv/Snp500_Ru1000/{ticker.replace('.', '_')}.csv"
            data.to_csv(filepath)
            print(f"✅ 下载成功: {ticker}")
            return True
    except Exception as e:
        print(f"❌ {ticker} 下载出错: {e}")
    return False

def download_main():
    print("正在从 Snp500_Ru1000 表加载股票清单...")
    try:
        conn = sqlite3.connect('yahoo_data.db')
        query = "SELECT Yahoo_adj_Ticker_symbol FROM master"
        data_df = pd.read_sql(query, conn)
        conn.close()
        
        # 拿到股票清单
        stocks = data_df['Yahoo_adj_Ticker_symbol'].tolist()
        print(f"成功找到 {len(stocks)} 只股票")
            
    except Exception as e:
        print(f"读取 SQLite 数据库失败: {e}")
        return

    # 设置下载日期
    start_date = "1970-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # 开始下载
    for ticker in stocks:
        success = downloader(ticker, start_date, end_date)
        if not success:
            fail_download['Snp500_Ru1000'].append(ticker)
        time.sleep(0.5) # 稍微休息下，保护 GitHub 的网络

    print(f"下载结束！失败数: {len(fail_download['Snp500_Ru1000'])}")

if __name__ == '__main__':
    start_time = time.time()
    download_main()
    print(f"总耗时: {time.time() - start_time:.2f}秒")

