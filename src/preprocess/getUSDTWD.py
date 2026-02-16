import datetime

import investpy
import pandas as pd
import os
import requests

def Myinvestpy():
    """
    方法1: 使用investpy套件 (推薦)
    """

    TARGET = "THB"

    print(f"=== 方法1: 使用investpy套件抓取{TARGET}/TWD匯率資料 ===")

    # 修復 403 錯誤：添加自定義 headers 避免被 Investing.com 封鎖
    original_post = requests.post
    original_get = requests.get

    try:

        def patched_request(method):
            def wrapper(*args, **kwargs):
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers'].update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'txt/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                })
                return method(*args, **kwargs)
            return wrapper

        # 應用 monkey patch
        requests.post = patched_request(original_post)
        requests.get = patched_request(original_get)

        # 使用investpy抓取USD/TWD匯率資料
        print(f"正在從Inveting.com抓取{TARGET}/TWD匯率資料...")

        # 創建dataset資料夾
        os.makedirs('dataset', exist_ok=True)

        # 設定日期範圍
        start_date = "01/01/2000"  # investpy使用DD/MM/YYYY格式
        end_date = datetime.datetime.now().strftime("%d/%m/%Y")

        # 抓取USD/TWD歷史資料
        data = investpy.get_currency_cross_historical_data(
            currency_cross=f'{TARGET}/TWD',
            from_date=start_date,
            to_date=end_date,
        )

        # 處理資料格式
        df = pd.DataFrame({
            'Date': data.index.strftime('%Y-%m-%d'),
            'Close': data['Close']
        })

        # 重設索引
        df = df.reset_index(drop=True)

        print(f"成功抓取 {len(df)} 筆資料")
        print(f"資料範圍: {df['Date'].iloc[0]} 到 {df['Date'].iloc[-1]}")

        # 儲存為CSV
        filename = f"dataset/{TARGET}_TWD.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        print(f"資料已儲存至: {filename}")

        # 顯示預覽
        print("\n前5筆資料:")
        print(df.head())
        print("\n後5筆資料:")
        print(df.tail())

        # 恢復原始 requests 函數
        requests.post = original_post
        requests.get = original_get

        return df

    except Exception as e:
        # 恢復原始 requests 函數
        requests.post = original_post
        requests.get = original_get

        print(f"investpy方法發生錯誤: {str(e)}")
        print("可能原因: investpy套件可能需要更新或Investing.com網站結構改變")
        return None


if __name__ == "__main__":
    Myinvestpy()
