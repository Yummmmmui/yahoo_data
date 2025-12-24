import random
import threading
import os
import time
import datetime
import sqlite3
import pandas as pd
import requests
import yfinance as yf
from calendar import monthrange
from util.database_postgresql import save_data_to_db, get_last_date_from_db, create_table_if_not_exists

print(f"yfinance 版本: {yf.__version__}")

# 定义全局变量 fail_download
fail_download = {'Shanghai_Shenzhen': [], 'Snp500_Ru1000': [], 'TSX': []}

def downloader(ticker, data_name, start_date, end_date, sleep_time=1, repeat=3, option=None, save_to_db=True, save_to_csv=True):
    """
    下载股票数据，支持yfinance和requests两种方式。
    """
    df_downloaded = None  

    for _ in range(repeat):
        if option is None or option == 0:
            try:
                data = yf.Ticker(ticker).history(
                    period="max",
                    interval="1d",
                    start=start_date,
                    end=end_date,
                    prepost=False,
                    actions=False,
                    auto_adjust=False,
                    back_adjust=False,
                    proxy=None,
                    rounding=False
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
                today = datetime.date.today()
                if today.month == 1:
                    target_month, target_year = 12, today.year - 1
                else:
                    target_month, target_year = today.month - 1, today.year
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
        if save_to_db:
            save_data_to_db(df_downloaded.copy(), ticker, data_name)
            print(f"DB: {ticker} Successful")
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

def active_downloader_threads_count():
    return sum(1 for t in threading.enumerate() if isinstance(t, threading.Thread) and t.is_alive() and hasattr(t, '_target') and t._target == downloader)

max_threads = 20

def download(data_option=0, use_threads=1, sleep_time=1, repeat=3, download_option_method=None, save_to_db=True, save_to_csv=True):
    if save_to_db:
        create_table_if_not_exists()

    # 准备目录
    os.makedirs('new_csv', exist_ok=True)
    for country in ['Shanghai_Shenzhen', 'Snp500_Ru1000', 'TSX']:
        os.makedirs(os.path.join('new_csv', country), exist_ok=True)

    print("正在从 SQLite 加载股票清单...")
    try:
        conn = sqlite3.connect('yahoo_data.db')
        # 根据您的 sqlite 表结构查询数据
        query = "SELECT country, Yahoo_adj_Ticker_symbol, [currently use] FROM master"
        all_data = pd.read_sql(query, conn)
        conn.close()
        
        # 根据 data_option 筛选数据
        sheet_map = {1: 'Shanghai_Shenzhen', 2: 'Snp500_Ru1000', 3: 'TSX'}
        if data_option in sheet_map:
            target_country = sheet_map[data_option]
            data = all_data[all_data['country'] == target_country].copy()
        else:
            data = all_data.copy()
            
    except Exception as e:
        print(f"读取 SQLite 数据库失败: {e}")
        return

    # 日期设置
    start_date = datetime.datetime(1970, 2, 1).date()
    now = datetime.datetime.now()
    if now.month == 1:
        end = datetime.datetime(now.year - 1, 12, monthrange(now.year - 1, 12)[1])
    else:
        end = datetime.datetime(now.year, now.month, 1) - datetime.timedelta(days=1)
    
    end_date = end
    endDate = end.strftime('%Y-%m-%d')
    thread_list = []

    for r in range(repeat):
        print(f"正在开始第 {3-repeat+1} 轮数据下载")
        for index, row in data.iterrows():
            ticker = row['Yahoo_adj_Ticker_symbol']
            data_name = row['country']
            if row['currently use'] != 'yes':
                continue

            # 检查是否需要更新
            data_is_up_to_date = False
            if save_to_db:
                last_db_date = get_last_date_from_db(ticker)
                if last_db_date == endDate:
                    data_is_up_to_date = True

            if data_is_up_to_date:
                continue

            if use_threads == 1:
                while active_downloader_threads_count() >= max_threads:
                    time.sleep(1)
                t = threading.Thread(target=downloader, args=(ticker, data_name, start_date, end_date),
                                     kwargs={'option': download_option_method, 'save_to_db': save_to_db, 'save_to_csv': save_to_csv})
                thread_list.append(t)
                t.start()
                time.sleep(sleep_time)
            else:
                downloader(ticker, data_name, start_date, end_date, option=download_option_method, save_to_db=save_to_db, save_to_csv=save_to_csv)

        for t in thread_list:
            t.join()
        thread_list.clear()

    # 记录失败
    record = ''
    for country in ['Shanghai_Shenzhen', 'Snp500_Ru1000', 'TSX']:
        record += f"\n{country}\n失败数量: {len(fail_download[country])}\n" + "\n".join(fail_download[country]) + "\n"
    
    os.makedirs('failed_txt', exist_ok=True)
    with open('failed_txt/failed.txt', 'w', encoding='utf-8') as f:
        f.write(record)

if __name__ == '__main__':
    start_time = time.time()
    # 0:全部, 1:上海_深圳, 2:标普500_罗素1000, 3:多伦多
    download(data_option=2, use_threads=0, sleep_time=1, repeat=1, download_option_method=1)
    print(f"总耗时: {time.time() - start_time:.2f}秒")
