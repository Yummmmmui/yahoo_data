import random
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
fail_download = {'Shanghai_Shenzhen': [], 'Snp500_Ru1000': [], 'TSX': []}

def downloader(ticker, data_name, start_date, end_date, sleep_time=1, repeat=3, option=None, save_to_csv=True):
    """
    下载股票数据，仅支持 yfinance 和 requests 两种方式，移除数据库保存功能。
    """
    df_downloaded = None  

    for _ in range(repeat):
        if option is None or option == 0:
            try:
                data = yf.Ticker(ticker).history(
                    period="max", interval="1d",
                    start=start_date, end=end_date,
                    auto_adjust=False, actions=False
                )
                if data is None or data.shape[0] <= 1:
                    print(ticker + ' yfinance returned no data')
                else:
                    data_monthly = data.asfreq('ME', method='pad')
                    data_monthly_db = data_monthly.reset_index()
                    data_monthly_db['Date'] = data_monthly_db['Date'].dt.strftime('%Y-%m-%d')
                    df_downloaded = data_monthly_db
                    break 

            except Exception as e:
                print(f"{ticker} yfinance error: {e}")
                time.sleep(sleep_time)

        if (option is None or option == 1) and df_downloaded is None:
            try:
                # 备选 API 下载逻辑保持不变
                today = datetime.date.today()
                target_month, target_year = (12, today.year - 1) if today.month == 1 else (today.month - 1, today.year)
                target_day = monthrange(target_year, target_month)[1]
                req_end_date = datetime.date(target_year, target_month, target_day)
                
                start_unix = int(time.mktime(start_date.timetuple()))
                end_unix = int(time.mktime((req_end_date + datetime.timedelta(days=1)).timetuple())) - 1

                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_unix}&period2={end_unix}&interval=1d&events=div%2Csplits"
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    result_data = result.get("chart", {}).get("result", [])
                    if result_data:
                        timestamps = result_data[0].get("timestamp", [])
                        indicators = result_data[0].get("indicators", {})
                        quote = indicators.get("quote", [{}])[0]
                        adjclose_data = indicators.get("adjclose", [{}])[0]

                        df = pd.DataFrame({
                            "Date": pd.to_datetime([datetime.datetime.fromtimestamp(ts, datetime.UTC) for ts in timestamps]).tz_localize(None),
                            "Open": quote.get("open", []),
                            "High": quote.get("high", []),
                            "Low": quote.get("low", []),
                            "Close": quote.get("close", []),
                            "Adj Close": adjclose_data.get("adjclose", []),
                            "Volume": quote.get("volume", []),
                        })

                        monthly_rows = []
                        for ym, group in df.groupby(df['Date'].dt.to_period('M')):
                            last_day = ym.to_timestamp(how='end')
                            last_valid_row = group.loc[group['Close'].notna()].iloc[-1]
                            new_row = last_valid_row.copy()
                            new_row['Date'] = last_day
                            monthly_rows.append(new_row)

                        df_monthly = pd.DataFrame(monthly_rows)
                        df_monthly['Date'] = df_monthly['Date'].dt.strftime('%Y-%m-%d')
                        df_downloaded = df_monthly
                        break
            except Exception as e_req:
                print(f"{ticker} requests API error: {e_req}")
                time.sleep(sleep_time)

    if df_downloaded is not None and df_downloaded.shape[0] > 1:
        if save_to_csv:
            os.makedirs(os.path.join('new_csv', data_name), exist_ok=True)
            filepath = os.path.join('new_csv', data_name, f"{ticker}.csv")
            cols_to_save = [col for col in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in df_downloaded.columns]
            df_csv = df_downloaded[cols_to_save].sort_values(by='Date', ascending=True)
            df_csv.to_csv(filepath, index=False)
            print(f"CSV: {ticker} Successful")
        return

    fail_download[data_name].append(ticker)
    print(ticker + ' Download Failed')

def download(data_option=0, sleep_time=1, repeat=1, download_option_method=None, save_to_csv=True):
    # 准备目录
    os.makedirs('new_csv', exist_ok=True)
    for country in ['Shanghai_Shenzhen', 'Snp500_Ru1000', 'TSX']:
        os.makedirs(os.path.join('new_csv', country), exist_ok=True)

    print("正在从 SQLite 加载股票清单...")
    try:
        conn = sqlite3.connect('yahoo_data.db')
        query = "SELECT Yahoo_adj_Ticker_symbol FROM Snp500_Ru1000"
        df = pd.read_sql(query, conn)
        conn.close()
        
        sheet_map = {1: 'Shanghai_Shenzhen', 2: 'Snp500_Ru1000', 3: 'TSX'}
        if data_option in sheet_map:
            target_country = sheet_map[data_option]
            data = all_data[all_data['country'] == target_country].copy()
        else:
            data = all_data.copy()
            
    except Exception as e:
        print(f"读取 SQLite 数据库失败: {e}")
        return

    start_date = datetime.datetime(1970, 2, 1).date()
    end_date = datetime.datetime.now() # 这里简化为当前时间

    for r in range(repeat):
        print(f"正在开始第 {r+1} 轮数据下载")
        for index, row in data.iterrows():
            ticker = row['Yahoo_adj_Ticker_symbol']
            data_name = row['country']
            if row['currently use'] != 'yes':
                continue

            downloader(ticker, data_name, start_date, end_date, option=download_option_method, save_to_csv=save_to_csv)

    # 记录失败情况
    record = ''
    for country in ['Shanghai_Shenzhen', 'Snp500_Ru1000', 'TSX']:
        record += f"\n{country}\n失败数量: {len(fail_download[country])}\n" + "\n".join(fail_download[country]) + "\n"
    
    os.makedirs('failed_txt', exist_ok=True)
    with open('failed_txt/failed.txt', 'w', encoding='utf-8') as f:
        f.write(record)

if __name__ == '__main__':
    start_time = time.time()
    download(data_option=2, sleep_time=1, repeat=1, download_option_method=1)
    print(f"总耗时: {time.time() - start_time:.2f}秒")


