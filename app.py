# ====必要なライブラリのインポート====

# streamlit(WebアプリのUI作成)
import streamlit as st
# データ操作ライブラリ(表形式データの処理)
import pandas as pd
import numpy as np
# 価格データの取得(Yahoo Finance API)
import yfinance as yf
# 日本語テキストの正規化
import unicodedata
# Webデータ取得関連
import requests
import re
from bs4 import BeautifulSoup
from io import StringIO
# 数学・統計処理(最小分散フロンティアの計算)
from scipy.optimize import minimize
# 日付操作(前営業日の取得など)
from datetime import datetime, date, timedelta
# プロット用ライブラリ
import matplotlib.pyplot as plt
# グラフ描画用ライブラリ
import plotly.graph_objs as go
# CSVテンプレート生成などに使用
import io
# 日本の祝日判定ライブラリ(jpholiday)
import jpholiday
# 時間操作のモジュール
import time
# 並列処理用
import concurrent.futures

# ====言語設定====

# 言語辞書
LANGUAGES = {
    "en": {
        "page_title": "Minimum Variance Frontier Calculation",
        "app_title": "Minimum Variance Frontier Calculation",
        "data_input_method": "Select Data Input Method",
        "ticker_search_input": "Input via Ticker Search",
        "csv_input": "Input via CSV",
        "ticker_search": "Ticker Search",
        "ticker_search_placeholder": "e.g., 7203, Toyota Motor, AAPL, Apple",
        "ticker_search_label": "Enter ticker code or name",
        "convert_usd_to_jpy": "Convert tickers not ending with '.T' from USD to JPY for analysis",
        "search_results": "Search Results",
        "results_count": "件",
        "add_button": "Add",
        "already_selected": "✓ Selected",
        "delete_button": "Delete",
        "reset_button": "Reset",
        "no_results": "No matching tickers found.",
        "selected_tickers": "Selected Tickers",
        "analysis_target_count": "Number of tickers for analysis",
        "tickers": "tickers",
        "analysis_period": "Analysis Period Settings",
        "start_date": "Start Date",
        "end_date": "End Date",
        "span_label": "Span (Daily, Weekly, Monthly)",
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "csv_file_upload": "Select CSV File",
        "sample_csv_download": "Download Sample CSV",
        "csv_preview": "Show CSV Preview",
        "span_auto_detected": "Span auto-detected",
        "avg_interval": "average interval",
        "days": "days",
        "analysis_params": "Analysis Parameters",
        "min_weight": "Minimum Investment Ratio",
        "num_steps": "Number of Expected Return Steps",
        "risk_free_rate_label": "Annual Rate of Risk-Free Asset (%)",
        "jgb_rate_info": "Short-term JGB rate",
        "jgb_rate_failed": "Failed to retrieve short-term JGB rate.",
        "market_portfolio": "Market Portfolio Selection",
        "market_portfolio_label": "Select Market Portfolio",
        "calc_button": "Execute Calculation",
        "calculating": "Calculating...",
        "analysis_results": "Analysis Results",
        "price_time_series": "Show Price Time Series (Closing Prices)",
        "fx_rate_display": "Show Exchange Rate (USD/JPY)",
        "fx_rate_label": "Exchange Rate (Yen)",
        "fx_rate_download": "Download Exchange Rate as CSV",
        "std_dev_return": "Show Standard Deviation and Expected Return for Each Ticker",
        "ticker_code": "Ticker Code",
        "std_dev": "Standard Deviation",
        "expected_return": "Expected Return",
        "correlation_matrix": "Show Correlation Matrix (Heatmap)",
        "correlation_download": "Download Correlation Matrix as CSV",
        "mvf_cml_display": "Show Minimum Variance Frontier (MVF) and Capital Market Line (CML)",
        "mvf_cml_download": "Download Standard Deviation and Investment Ratio for Each Expected Return as CSV",
        "sml_display": "Show Security Market Line (SML)",
        "beta": "β",
        "beta_download": "Download β Values for Each Ticker as CSV",
        "beta_value": "β Value",
        "disclaimer": "This app is created for educational purposes and is not intended for investment decisions.<br>The developer assumes no responsibility for any damages arising from the use of this app.",
        "error_future_date": "End date is in the future. Please select a valid end date.",
        "error_date_order": "Start date must be earlier than end date.",
        "error_min_tickers": "You need to add at least 2 tickers to the list.",
        "error_price_fetch": "Price data fetch error",
        "error_price_empty": "Failed to fetch price data.",
        "error_csv_date_format": "Invalid date format.",
        "error_csv_processing": "Error occurred during CSV data processing",
        "error_log_return": "Failed to calculate log returns",
        "error_valid_tickers": "At least 2 tickers with valid data are required. Please review the period.",
        "info_common_data": "For the specified period",
        "info_common_data_only": "only",
        "info_common_data_available": "of common valid price data is available.",
        "error_min_weight_large": "Minimum investment ratio is too large for the selected number of tickers (must be less than",
        "error_min_weight_range": "Minimum investment ratio must be between 0 and 0.5.",
        "error_mvf_failed": "Failed to calculate minimum variance frontier. Please review the number of tickers, period, and minimum investment ratio.",
        "error_date_not_set": "Dates are not set correctly.",
        "error_market_data_failed": "Failed to fetch market portfolio data for",
        "error_market_data_fetch": "Error occurred while fetching market data",
        "error_no_price_data": "Price data does not exist.",
        "error_fx_display": "Failed to display exchange rate",
        "error_correlation_display": "Error occurred while displaying correlation matrix",
        "error_mvf_not_calculated": "Minimum variance frontier could not be calculated correctly.",
        "warning_correlation_data": "Price data must be fetched first to display correlation matrix.",
        "error_csv_no_data": "CSV file data not found.",
        "error_span_estimation": "Error occurred during span estimation",
        "error_column_date_format": "Failed to format column names to date display",
        "error_invalid_market_portfolio": "Invalid market portfolio selected.",
        "date_label": "Date",
        "exchange_rate": "Exchange Rate",
        "language_select": "Language",
        "warning_jp_stock_list_failed": "Failed to retrieve Japanese stock list",
        "warning_yahoo_search_error": "Yahoo Finance search error",
        "warning_jp_stock_search_error": "Japanese stock search error",
        "warning_data_fetch_error": "Data fetch error for",
        "warning_result_fetch_error": "Result fetch error for",
        "warning_fx_conversion_failed": "Failed to convert currency",
        "error_no_valid_price_data": "Failed to retrieve any valid price data.",
        "error_no_valid_tickers": "No valid tickers provided."
    },
    "ja": {
        "page_title": "最小分散フロンティアの計算",
        "app_title": "最小分散フロンティアの計算",
        "data_input_method": "データ入力方法を選択",
        "ticker_search_input": "銘柄検索による入力",
        "csv_input": "CSVによる入力",
        "ticker_search": "銘柄検索",
        "ticker_search_placeholder": "例: 7203, トヨタ自動車, AAPL, Apple",
        "ticker_search_label": "銘柄コードまたは銘柄名を入力",
        "convert_usd_to_jpy": "証券コードの末尾が'.T'以外の銘柄をドル円換算して分析する",
        "search_results": "検索結果",
        "results_count": "件",
        "add_button": "追加",
        "already_selected": "✓ 選択済み",
        "delete_button": "削除",
        "reset_button": "リセット",
        "no_results": "該当する銘柄が見つかりませんでした．",
        "selected_tickers": "選択中の銘柄リスト",
        "analysis_target_count": "分析対象銘柄数",
        "tickers": "銘柄",
        "analysis_period": "分析期間の設定",
        "start_date": "開始日",
        "end_date": "終了日",
        "span_label": "スパン（日足・週足・月足）",
        "daily": "日足",
        "weekly": "週足",
        "monthly": "月足",
        "csv_file_upload": "CSVファイルを選択",
        "sample_csv_download": "サンプルCSVをDL",
        "csv_preview": "CSVのプレビューを表示",
        "span_auto_detected": "スパンは自動判定されました",
        "avg_interval": "平均間隔",
        "days": "日",
        "analysis_params": "分析パラメータの設定",
        "min_weight": "最小投資割合",
        "num_steps": "期待利益率の段階数",
        "risk_free_rate_label": "無リスク資産の年利（%）",
        "jgb_rate_info": "の短期国債金利は",
        "jgb_rate_failed": "短期国債金利の取得に失敗しました．",
        "market_portfolio": "比較市場ポートフォリオの選択",
        "market_portfolio_label": "市場ポートフォリオを選択",
        "calc_button": "計算を実行",
        "calculating": "計算中です...",
        "analysis_results": "分析結果",
        "price_time_series": "価格の時系列（終値）を表示",
        "fx_rate_display": "為替レート（USD/JPY）の表示",
        "fx_rate_label": "為替レート（円）",
        "fx_rate_download": "為替レートをCSVとしてダウンロード",
        "std_dev_return": "各銘柄の標準偏差と期待利益率を表示",
        "ticker_code": "証券コード",
        "std_dev": "標準偏差",
        "expected_return": "期待利益率",
        "correlation_matrix": "銘柄間の相関行列（ヒートマップ）を表示",
        "correlation_download": "相関係数をCSVとしてダウンロード",
        "mvf_cml_display": "最小分散フロンティア(MVF)と資本市場線(CML)を表示",
        "mvf_cml_download": "各期待利益率における標準偏差と投資割合をCSVとしてダウンロード",
        "sml_display": "証券市場線(SML)を表示",
        "beta": "β",
        "beta_download": "各銘柄のβ値をCSVとしてダウンロード",
        "beta_value": "β値",
        "disclaimer": "本アプリは学習目的で作成されたものであり，投資判断への利用を想定したものではありません．<br> 本アプリの利用によって生じたいかなる損害についても開発者は責任を負いかねます．",
        "error_future_date": "終了日が未来の日付になっています．正しい終了日を選んでください．",
        "error_date_order": "開始日は終了日より前の日付を選択してください．",
        "error_min_tickers": "2銘柄以上をリストに追加する必要があります．",
        "error_price_fetch": "価格データ取得エラー",
        "error_price_empty": "価格データの取得に失敗しました．",
        "error_csv_date_format": "日付の形式が不正です．",
        "error_csv_processing": "CSVのデータ処理中にエラーが発生しました",
        "error_log_return": "ログリターンの計算に失敗しました",
        "error_valid_tickers": "有効なデータを持つ銘柄が2つ以上必要です．期間を見直してください．",
        "info_common_data": "指定された期間（",
        "info_common_data_only": "）に対し，共通の有効価格データが存在するのは ",
        "info_common_data_available": " のみです．",
        "error_min_weight_large": "選択された銘柄数に対して最小投資割合が大きすぎます（",
        "error_min_weight_range": "最小投資割合は0以上0.5未満である必要があります．",
        "error_mvf_failed": "最小分散フロンティアの計算に失敗しました．銘柄数・期間・最小投資割合を見直してください．",
        "error_date_not_set": "日付が正しく設定されていません．",
        "error_market_data_failed": "市場ポートフォリオ",
        "error_market_data_fetch": "市場データ取得中にエラーが発生しました",
        "error_no_price_data": "価格データが存在しません．",
        "error_fx_display": "為替レートの表示に失敗しました",
        "error_correlation_display": "相関行列の表示中にエラーが発生しました",
        "error_mvf_not_calculated": "最小分散フロンティアが正常に計算できませんでした．",
        "warning_correlation_data": "相関行列を表示するには先に価格データの取得が必要です．",
        "error_csv_no_data": "CSVファイルのデータが見つかりません．",
        "error_span_estimation": "スパンの推定中にエラーが発生しました",
        "error_column_date_format": "列名の日付表示の整形に失敗しました",
        "error_invalid_market_portfolio": "無効な市場ポートフォリオが選択されました．",
        "date_label": "日付",
        "exchange_rate": "為替レート",
        "language_select": "言語",
        "warning_jp_stock_list_failed": "日本銘柄リストの取得に失敗しました",
        "warning_yahoo_search_error": "Yahoo Finance検索エラー",
        "warning_jp_stock_search_error": "日本銘柄検索エラー",
        "warning_data_fetch_error": "のデータ取得エラー",
        "warning_result_fetch_error": "の結果取得エラー",
        "warning_fx_conversion_failed": "為替換算に失敗しました",
        "error_no_valid_price_data": "有効な価格データが一つも取得できませんでした．",
        "error_no_valid_tickers": "有効なティッカーが1つもありません．"
    }
}

# ====初期化====

_data = None  # 終値のDaraFrame
rf_rate_span = None  # 無リスク利子率(スパン単位換算値)

# ====ページ設定とカスタムスタイル====

# Webアプリのタイトルとレイアウト方向を設定
st.set_page_config(
    page_title="最小分散フロンティアの計算",  # ブラウザのタブに表示されるタイトル
    layout="wide"  # 表示の中央寄せ
)
# アプリの背景やボタン等の見た目をCSSで変更
st.markdown("""
    <style>
    .block-container {
        background-color: #000000;  /* 背景を黒に */
        color: #ffffff;  /* 文字色を白に */
        font-family: Meiryo, sans-serif;  /* メイリオフォントを使用 */
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 100% !important;
    }
    /* ボタンの見た目とレイアウト調整 */
    div.stButton > button:first-child {
        width: 100% !important;
        text-align: center !important;
    }
    .template-button-container {
        display: flex;
        justify-content: flex-end;
        margin-top: -2rem;
        margin-bottom: 1rem;
    }
    .small-button button {
        font-size: 0.75rem !important;
        padding: 0.2rem 0.5rem !important;
    }
    /* 不要なヘッダー・フッター・デプロイボタン等を非表示 */
    header, footer, .stActionButton, .stDeployButton, .st-emotion-cache-13ln4jf, .st-emotion-cache-1avcm0n {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# ====セッションステートの初期化====

# Streamlitアプリでは，セッションごとに変数を保持できる「st.session_state」を使って状態管理を行う．
# 各変数が未定義の場合に初期値を設定する．
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # デフォルト言語は英語
if 'calculating' not in st.session_state:
    st.session_state.calculating = False  # 計算処理中であることを表すフラグ
if 'result_data' not in st.session_state:
    st.session_state.result_data = None  # 計算結果を格納する変数
if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = []  # ユーザが選択した銘柄のリスト

# 言語取得関数
def t(key):
    """言語辞書からテキストを取得"""
    return LANGUAGES[st.session_state.language].get(key, key)


# ====財務省から短期国債金利を取得する関数====

def get_latest_jgb_1year_rate():
    """財務省から短期国債金利の最新データを取得"""
    try:
        csv_url = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        csv_response = requests.get(csv_url, headers=headers, timeout=10)
        csv_response.raise_for_status()
        
        # 複数のエンコーディングを試行
        for encoding in ['shift_jis', 'utf-8', 'cp932']:
            try:
                csv_content = csv_response.content.decode(encoding)
                df = pd.read_csv(StringIO(csv_content), header=None)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            return None
        
        # 有効なデータを持つ行を抽出
        valid_rows = df[df.iloc[:, 1].notna()]
        if valid_rows.empty:
            return None
        
        # 最新の有効データを取得
        last_valid_index = valid_rows.index[-1]
        latest_date = df.iloc[last_valid_index, 0]
        latest_rate = df.iloc[last_valid_index, 1]
        
        # 利率を数値に変換
        try:
            latest_rate = float(latest_rate)
        except (ValueError, TypeError):
            return None
        
        return csv_url, latest_date, latest_rate, df
        
    except Exception as e:
        # エラーログは呼び出し側で処理
        return None


# ====日本の銘柄リストを取得する関数====

@st.cache_data
def load_japan_stock_list():
    """
    日本取引所(JPX)が公開している銘柄リスト(Excel)を読み込み，
    証券コードと銘柄名の一覧をDataFrameとして返す．
    読み込みに失敗した場合はNoneを返す．
    """
    # 東京証券取引所が提供する東証上場銘柄一覧(Excel)
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        # Excelを読み込む(1行目をスキップ)
        df = pd.read_excel(url, skiprows=1, header=None)
        # 必要な列(1列目:コード，2列目:銘柄名)のみ抽出し，欠損行を除外
        df = df[[1, 2]].dropna()
        # 列名を設定(コードと銘柄名)
        df.columns = ['コード', '銘柄名']
        # コードを文字列に変換して空白除去
        df['コード'] = df['コード'].astype(str).str.strip()
        return df
    except Exception as e:
        # エラー発生時は警告表示し，Noneを返す
        st.warning(f"{t('warning_jp_stock_list_failed')}: {e}")
        return None


# ====Yahoo Finance検索APIと日本銘柄リストを使った銘柄検索クラス====

class AssetSearcher:
    def __init__(self, jp_stock_df=None):
        self.search_url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.last_request_time = 0
        self.min_request_interval = 0.5  # レート制限（0.5秒）
        self.jp_stock_df = jp_stock_df  # 日本銘柄リスト
    
    def search_assets(self, query, max_results=20):
        """銘柄を検索してリストを返す"""
        if not query or len(query.strip()) < 1:
            return []
        
        query = query.strip()
        assets = []
        
        # 1. まず日本銘柄リストから検索（日本語対応）
        if self.jp_stock_df is not None:
            jp_results = self._search_japan_stocks(query)
            assets.extend(jp_results)
        
        # 2. Yahoo Finance APIで検索（英数字）
        try:
            self._rate_limit()
            yahoo_results = self._call_yahoo_search_api(query)
            
            for result in yahoo_results:
                asset = self._convert_to_asset_info(result)
                if asset:
                    # 重複を避ける（同じシンボルが既にある場合はスキップ）
                    if not any(a['symbol'] == asset['symbol'] for a in assets):
                        assets.append(asset)
        except Exception as e:
            st.warning(f"{t('warning_yahoo_search_error')}: {e}")
        
        # 最大結果数に制限
        return assets[:max_results]
    
    def _search_japan_stocks(self, query):
        """日本銘柄リストから検索（証券コードまたは銘柄名）"""
        results = []
        
        try:
            # 証券コードで検索
            code_matches = self.jp_stock_df[
                self.jp_stock_df['コード'].str.contains(query, case=False, na=False)
            ]
            
            # 銘柄名で検索（日本語対応）
            name_matches = self.jp_stock_df[
                self.jp_stock_df['銘柄名'].str.contains(query, case=False, na=False)
            ]
            
            # 結果をマージ（重複を除く）
            matches = pd.concat([code_matches, name_matches]).drop_duplicates()
            
            # 最大20件に制限
            for _, row in matches.head(20).iterrows():
                results.append({
                    'symbol': row['コード'] + '.T',  # .Tを付けて東証銘柄として識別
                    'name': row['銘柄名'],
                    'exchange': 'Tokyo',
                    'currency': 'JPY',
                    'type': 'EQUITY'
                })

        except Exception as e:
            st.warning(f"{t('warning_jp_stock_search_error')}: {e}")
        
        return results
    
    def _rate_limit(self):
        """レート制限を適用"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _call_yahoo_search_api(self, query):
        """Yahoo Finance検索APIを呼び出す"""
        try:
            params = {'q': query, 'quotesCount': 15, 'newsCount': 0}
            response = self.session.get(self.search_url, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            return data.get('quotes', [])
        except Exception as e:
            # エラーは警告として表示せず，空のリストを返す
            return []
    
    def _convert_to_asset_info(self, yahoo_result):
        """Yahoo Financeの検索結果を資産情報に変換"""
        try:
            symbol = yahoo_result.get('symbol')
            if not symbol:
                return None
            
            name = (yahoo_result.get('longname') or 
                   yahoo_result.get('shortname') or 
                   symbol)
            
            exchange = yahoo_result.get('exchange', '')
            currency = yahoo_result.get('currency', 'USD')
            quote_type = yahoo_result.get('quoteType', '')
            
            return {
                'symbol': symbol,
                'name': name,
                'exchange': exchange,
                'currency': currency,
                'type': quote_type
            }
            
        except Exception as e:
            return None


# ====ユーザー入力の正規化関数====

def normalize_input(text):
    """
    入力文字列を正規化(全角→半角，記号の統一など)して返す．
    例: '７２０３' → '7203'
    """
    return unicodedata.normalize('NFKC', text)


# ====言語選択UI====

# 言語選択を右上に配置
col1, col2 = st.columns([4, 1])
with col2:
    language_options = {"English": "en", "日本語": "ja"}
    selected_language_name = st.selectbox(
        t("language_select"),
        options=list(language_options.keys()),
        index=0 if st.session_state.language == "en" else 1,
        key="language_selector"
    )
    # 言語が変更された場合、セッションステートを更新
    new_language = language_options[selected_language_name]
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.rerun()

# ====タイトル表示(カスタムCSSで装飾)====

# 通常表示では大きく，モバイル表示では小さく表示されるように調整
st.markdown(f"""
    <style>
    /* タイトルの基本スタイル */
    .custom-title {{
        text-align: center;
        font-size: 40px;
        margin-bottom: 1rem;
    }}
    /* モバイル表示（画面幅600px以下）ではフォントサイズを縮小 */
    @media screen and (max-width: 600px) {{
        .custom-title {{
            font-size: 25px;
        }}
    }}
    </style>
    <!-- タイトルを表示する要素 -->
    <div class='custom-title'>{t("app_title")}</div>
""", unsafe_allow_html=True)

# 区切り線を表示
st.markdown("""
<hr style='border: 1px solid white; margin: 25px 0;' />
""", unsafe_allow_html=True)


# ====入力方式選択====

# ユーザにデータ入力の方式(CSV or 銘柄検索)を選ばせる
# 前回の入力方式をセッションから取得(初回はNone)
previous_input_mode = st.session_state.get("previous_input_mode", None)
# ユーザに入力方式を選ばせるラジオボタン
input_mode = st.radio(
    t("data_input_method"),
    [t("ticker_search_input"), t("csv_input")],
    horizontal=False
)
# 入力方式が前回と異なる場合，セッション情報を初期化(結果などをクリア)
if "previous_input_mode" not in st.session_state:
    st.session_state.previous_input_mode = input_mode
if st.session_state.previous_input_mode != input_mode:
    st.session_state._df = None
    st.session_state.df_csv = None
    st.session_state.log_returns = None
    st.session_state.result_data = None
    st.session_state.calculating = False
    st.session_state.previous_input_mode = input_mode
# 現在の入力方式をセッションに記録(次回の比較に使用)
st.session_state.previous_input_mode = input_mode
# CSV入力モードを真偽値で保持(以降の分岐処理に使用)
use_csv = input_mode == t("csv_input")
st.session_state.use_csv = use_csv

# 区切り線を表示
st.markdown("---")


# ====入力データ関連の初期化====

uploaded_file = None  # ユーザがアップロードするCSVファイルを格納
log_returns = None  # 価格データから計算されるログリターン(DataFrameまたは配列)


# ====ドル円レートを取得する関数====

@st.cache_data
def fetch_usd_to_jpy_rates(start_date, end_date, interval):
    """
    指定期間・スパンでドル円為替レートを取得する（USD/JPY）．
    """
    fx = yf.download("JPY=X", start=start_date, end=end_date, interval=interval, progress=False)
    if fx is None or fx.empty or "Close" not in fx.columns:
        raise ValueError("ドル円為替レートの取得に失敗しました．")
    fx = fx["Close"]
    if fx.index.tz is not None:
        fx.index = fx.index.tz_convert("Asia/Tokyo").tz_localize(None)
    return fx.sort_index()


# ====終値の取得関数====

def fetch_single_asset_data(args):
    """単一銘柄のデータを取得（並列処理用）"""
    symbol, start_date, end_date, interval = args
    
    try:
        # yfinanceでデータ取得（auto_adjust=Falseに設定）
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start_date,
            end=end_date + timedelta(days=1),  # 終了日を含むため+1日
            interval=interval,
            auto_adjust=False,  # Adj Close取得に必須
            prepost=False,      # プレ・ポストマーケットを除外
            repair=True         # データの修復を有効化
        )
        
        if not hist.empty and 'Adj Close' in hist.columns:
            # 調整後終値を取得
            price_series = hist['Adj Close'].copy()
            price_series.name = symbol
            
            # タイムゾーン情報がある場合のみ削除
            if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
                price_series.index = price_series.index.tz_localize(None)
            
            # 日付のみに正規化（時間部分を削除）
            price_series.index = price_series.index.normalize()
            
            # NaN値を除去（補完は行わない）
            price_series = price_series.dropna()
            
            return {
                'symbol': symbol,
                'data': price_series,
                'success': True
            }
        else:
            return {
                'symbol': symbol,
                'data': pd.Series(dtype=float),
                'success': False
            }

    except Exception as e:
        st.warning(f"{t('warning_data_fetch_error')} {symbol}: {e}")
        return {
            'symbol': symbol,
            'data': pd.Series(dtype=float),
            'success': False,
            'error': str(e)
        }


def fetch_close_prices(symbols, start_date, end_date, interval):
    """
    与えられた証券コードリストに対して，指定期間・頻度の調整後終値を並列取得する．
    """
    if not symbols:
        raise ValueError(t("error_no_valid_tickers"))
    
    # 並列処理用の引数リストを作成
    args_list = [(symbol, start_date, end_date, interval) for symbol in symbols]
    
    price_data = {}
    
    # 並列処理でデータ取得
    max_workers = min(4, len(symbols))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 全てのタスクを投入
        future_to_symbol = {
            executor.submit(fetch_single_asset_data, args): args[0] 
            for args in args_list
        }
        
        # 完了したタスクから順次処理
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result['success'] and not result['data'].empty:
                    price_data[result['symbol']] = result['data']
            except Exception as e:
                st.warning(f"{t('warning_result_fetch_error')} {symbol}: {e}")

    if not price_data:
        raise ValueError(t("error_no_valid_price_data"))
    
    # DataFrameに統合
    df_merged = pd.DataFrame(price_data).sort_index()
    
    # 共通日付のみに絞る
    common_dates = df_merged.dropna().index
    df_merged = df_merged.loc[common_dates]
    
    # 為替換算オプションがオンなら，非JPY通貨をドル円換算（簡易的に全てUSDと仮定）
    if st.session_state.get("convert_usd_to_jpy", False):
        try:
            fx_rates = fetch_usd_to_jpy_rates(start_date, end_date, interval)
            if isinstance(fx_rates, pd.DataFrame):
                fx_rates = fx_rates.squeeze()
            
            # 米国銘柄を検出（.Tで終わらないものを米国銘柄と仮定）
            for symbol in df_merged.columns:
                if not symbol.endswith(".T"):  
                    aligned_fx = fx_rates.reindex(df_merged.index).ffill().bfill()
                    if len(aligned_fx) == len(df_merged.index):
                        df_merged[symbol] = df_merged[symbol] * aligned_fx
        except Exception as e:
            st.warning(f"{t('warning_fx_conversion_failed')}: {e}")
    
    return df_merged


# ====終値からログリターンを計算する関数====

def calculate_log_returns(df, axis="auto"):
    """
    指定された方向に沿ってログリターンを計算する．
    - axis=0 : 列方向（通常，行＝日付，列＝銘柄）
    - axis=1 : 行方向（通常，行＝銘柄，列＝日付）
    - axis='auto' の場合は列名が日付型なら axis=1，それ以外は axis=0 に自動判定
    """
    if df.isnull().values.any():
        raise ValueError("欠損値が含まれています．")
    if (df <= 0).values.any():
        raise ValueError("0以下の価格が含まれています．")
    if df.shape[1] < 2:
        raise ValueError("日付列が2列以上必要です．")
    df = df.sort_index()  # 日付順にソート
    # axis 自動判定：datetime型がどちらにあるかで決定
    if axis == "auto":
        if pd.api.types.is_datetime64_any_dtype(df.columns):
            axis = 1
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            axis = 0
        else:
            raise ValueError("時系列（日付）情報が index/columns のどちらにも見つかりません．")
    return np.log(df / df.shift(axis=axis)).dropna(axis=axis)


# ====CSV入力モード====

if use_csv:
    # span及びintervalを初期化
    span = None
    interval = None
    # CSVファイルのアップロード
    uploaded_file = st.file_uploader(t("csv_file_upload"), type="csv")
    # サンプルCSVの作成
    # 本日を含めた直近10営業日(平日(祝日は考慮しない))の日付のリストを作成
    sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=10, freq="B")
    # ダミーの価格データを作成
    template_data = pd.DataFrame({
        date.strftime("%Y-%m-%d"): [  # 各日付を"YYYY-MM-DD"形式にして列に
            np.random.randint(2500, 2750),  # 7203用(例:トヨタ自動車)
            np.random.randint(3250, 3750),  # 6758用(例:ソニーグループ)
            np.random.randint(140, 150)  # 9432用(例:日本電信電話)
         ] for date in sample_dates  # 各営業日毎に値を生成
     }, index=["7203.T", "6758.T", "9432.T"])  # 証券コードを行に
    # 1行目のインデックス名を空にする
    template_data.index.name = ""
    # CSV化(文字列としてメモリ上に保存)
    csv_buffer = io.StringIO()
    template_data.to_csv(csv_buffer)
    # CSV文字列を取り出す
    csv_data = csv_buffer.getvalue()
    # サンプルCSVのダウンロード
    st.download_button(t("sample_csv_download"), data=csv_data, file_name="sample.csv", mime="text/csv")
    # 実際のCSVファイル処理
    if uploaded_file:
        try:
            # CSVの読み込み(index_col=0で証券コードをインデックスに)
            df_csv = pd.read_csv(uploaded_file, index_col=0)
            # 日付列の変換(文字列→datetime)
            parsed_dates = pd.to_datetime(df_csv.columns, errors='coerce')
            if parsed_dates.isnull().any():
                raise ValueError(t("error_csv_date_format"))
            df_csv.columns = parsed_dates  # 日付型に変換
            # 必ず日付順にソート（念のため）
            df_csv = df_csv.sort_index(axis=1)
            # start_dateとend_dateの推定
            st.session_state.start_date = df_csv.columns.min().to_pydatetime().date()
            st.session_state.end_date = df_csv.columns.max().to_pydatetime().date()
            # スパン(日/週/月)の推定(平均間隔による)
            dates = df_csv.columns
            try:
                deltas = np.diff(dates).astype('timedelta64[D]').astype(int)
                avg_delta = np.mean(deltas)
                if avg_delta <= 2:
                    span = t("daily")
                    interval = "1d"
                elif avg_delta <= 10:
                    span = t("weekly")
                    interval = "1wk"
                else:
                    span = t("monthly")
                    interval = "1mo"
                st.session_state.interval = interval
                st.info(f"{t('span_auto_detected')}：**{span}**（{t('avg_interval')} {avg_delta:.1f} {t('days')}）")
            except Exception as e:
                st.error(f"{t('error_span_estimation')}：{e}")
                st.stop()
            # CSVプレビュー表示
            df_csv_display = df_csv.copy()
            try:
                df_csv_display.columns = [col.strftime('%Y/%m/%d') for col in df_csv_display.columns]
            except Exception as e:
                st.error(f"{t('error_column_date_format')}（{e}）．")
                st.stop()
            with st.expander(t("csv_preview")):
                st.dataframe(df_csv_display)
            num_csv_tickers = df_csv_display.shape[0]
            st.markdown(f"<p style='font-size: 16px; color: lightgray;'>{t('analysis_target_count')}：<strong>{num_csv_tickers}</strong> {t('tickers')}</p>", unsafe_allow_html=True)
            # 価格データをセッションに保存
            st.session_state.df_csv = df_csv  # 読み込んだ価格（行：銘柄，列：日付）
        except Exception as e:
            st.error(f"{t('error_csv_processing')}: {e}")


# ====銘柄検索入力モード====

if not use_csv:
    # span及びrf_rate_spanを初期化
    span = None
    rf_rate_span = None

    # 銘柄検索機能
    st.markdown(f"### {t('ticker_search')}")
    
    # 日本銘柄リストを読み込む
    if 'jp_stock_df' not in st.session_state:
        st.session_state.jp_stock_df = load_japan_stock_list()
    
    jp_stock_df = st.session_state.jp_stock_df
    
    # AssetSearcherのインスタンスを作成（セッション状態で管理）
    if 'asset_searcher' not in st.session_state:
        st.session_state.asset_searcher = AssetSearcher(jp_stock_df=jp_stock_df)
    
    searcher = st.session_state.asset_searcher
    
    # 検索入力
    search_query = normalize_input(st.text_input(
        t("ticker_search_label"),
        placeholder=t("ticker_search_placeholder"),
        key="search_input"
    ))

    # 為替換算オプション
    st.checkbox(t("convert_usd_to_jpy"), key="convert_usd_to_jpy")
    
    # 検索結果の表示
    if search_query:
        with st.spinner(t("calculating")):
            search_results = searcher.search_assets(search_query, max_results=20)

        if search_results:
            st.markdown(f"**{t('search_results')}: {len(search_results)}{t('results_count')}**")
            
            # 検索結果を選択肢として表示
            for i, asset in enumerate(search_results):
                col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
                
                with col1:
                    st.write(f"**{asset['symbol']}**")
                with col2:
                    st.write(f"{asset['name']}")
                with col3:
                    # 取引所と通貨の表示
                    exchange_info = asset['exchange'] if asset['exchange'] else 'N/A'
                    st.write(f"{asset['currency']} ({exchange_info})")
                with col4:
                    # 既に選択済みかチェック
                    is_selected = any(a['symbol'] == asset['symbol'] for a in st.session_state.selected_assets)

                    if not is_selected:
                        if st.button(t("add_button"), key=f"add_{i}_{asset['symbol']}"):
                            st.session_state.selected_assets.append(asset)
                            st.rerun()
                    else:
                        st.write(t("already_selected"))
        else:
            st.warning(t("no_results"))
    
    # 選択済み銘柄の表示
    if st.session_state.selected_assets:
        st.markdown("---")
        st.markdown(f"### {t('selected_tickers')}")
        num_selected = len(st.session_state.selected_assets)
        st.markdown(f"<p style='font-size: 16px; color: lightgray;'>{t('analysis_target_count')}：<strong>{num_selected}</strong> {t('tickers')}</p>", unsafe_allow_html=True)
        
        for i, asset in enumerate(st.session_state.selected_assets):
            cols = st.columns([2, 4, 2, 1])
            cols[0].write(asset["symbol"])
            cols[1].write(asset["name"])
            exchange_display = asset.get('exchange', 'N/A')
            cols[2].write(f"{asset['currency']} ({exchange_display})")
            if cols[3].button(t("delete_button"), key=f"del_{i}"):
                st.session_state.selected_assets.pop(i)
                st.rerun()

        if st.button(t("reset_button"), key="reset", type="secondary"):
            st.session_state.selected_assets = []
            st.session_state.result_data = None
            st.rerun()

        # 日付とスパンの設定
        st.markdown("---")
        st.markdown(f"### {t('analysis_period')}")
        
        def_date_end = date.today() - timedelta(days=1)
        def_date_start = def_date_end - timedelta(days=365)
        start_date = st.date_input(t("start_date"), value=def_date_start)
        end_date = st.date_input(t("end_date"), value=def_date_end)

        if end_date > date.today():
            st.error(t("error_future_date"))
            st.stop()
        elif start_date >= end_date:
            st.error(t("error_date_order"))
            st.stop()

        span = st.radio(t("span_label"), [t("daily"), t("weekly"), t("monthly")])
        interval_map = {t("daily"): "1d", t("weekly"): "1wk", t("monthly"): "1mo"}
        interval = interval_map[span]
        
        # 価格データ取得
        symbols = [a["symbol"] for a in st.session_state.selected_assets]

        if len(symbols) < 2:
            st.info(t("error_min_tickers"))
            st.stop()

        try:
            with st.spinner(t("calculating")):
                close_df = fetch_close_prices(symbols, start_date, end_date, interval)
        except ValueError as e:
            st.error(f"{t('error_price_fetch')}：{e}")
            st.stop()

        if close_df.empty:
            st.error(t("error_price_empty"))
            st.stop()
        
        # ソート（念のため）
        close_df = close_df.sort_index()
        
        # 価格DataFrameをセッションに保存
        st.session_state.close_df = close_df
        
        # 日付情報をセッションに保存
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.interval = interval


# ====最小投資割合と期待利益率の段階数の入力====

st.markdown("---")
st.markdown(f"### {t('analysis_params')}")

min_weight = st.number_input(t("min_weight"), min_value=0.0, max_value=0.5, value=0.00, step=0.001, format="%.3f")
num_steps = st.number_input(t("num_steps"), min_value=5, max_value=500, value=50, step=1)


# ====無リスク金利の取得と初期値の決定====

result = get_latest_jgb_1year_rate()
if result:  # 取得に成功した場合
    csv_url, latest_date, latest_rate, df = result
    if st.session_state.language == "ja":
        st.info(f"{latest_date} {t('jgb_rate_info')} {latest_rate:.3f}% です．\n[財務省]({csv_url})")
    else:
        st.info(f"{t('jgb_rate_info')} on {latest_date}: {latest_rate:.3f}%\n[Ministry of Finance]({csv_url})")
    rf_rate_default = latest_rate
else:  # 取得に失敗した場合
    st.warning(t("jgb_rate_failed"))
    rf_rate_default = 0.50  # デフォルト値

# ユーザが任意に設定可能な入力フォーム(%で表示)
rf_rate = st.number_input(
    t("risk_free_rate_label"),
    min_value=0.0, max_value=100.0,
    value=rf_rate_default,
    step=0.001,
    format="%.3f"
) / 100  # 実際の計算では少数を使う

# rf_rate(年利)をスパン単位に変換（定義のみ）
def convert_rf_rate_safe(rf_rate, span):
    """
    年率のリスクフリーレートをスパン（日足・週足・月足）に応じて変換．
    spanが不明な場合はエラーとして処理を停止．
    """
    daily_label = t("daily")
    weekly_label = t("weekly")
    monthly_label = t("monthly")

    if span == daily_label:
        return rf_rate / 252
    elif span == weekly_label:
        return rf_rate / 52
    elif span == monthly_label:
        return rf_rate / 12
    else:
        st.error(f"{t('error_date_not_set')}（{span}）")
        st.stop()

# span(日/週/月)が決まったら，rf_rate(年利)をスパン単位に変換
if span:
    rf_rate_span = convert_rf_rate_safe(rf_rate, span)


# ====比較市場ポートフォリオの選択====

st.markdown("---")
st.markdown(f"### {t('market_portfolio')}")

# 市場ポートフォリオ選択
market_choice = st.radio(
    t("market_portfolio_label"),
    (
        "Nikkei 225 (^N225)",
        "NASDAQ Composite (^IXIC)",
        "S&P 500 (^GSPC)",
        "Dow Jones Industrial Average (^DJI)"
    )
)

# 選択肢からティッカーコードを割り出す
if "Nikkei 225 (^N225)" in market_choice:
    market_ticker = "^N225"
elif "NASDAQ Composite (^IXIC)" in market_choice:
    market_ticker = "^IXIC"
elif "S&P 500 (^GSPC)" in market_choice:
    market_ticker = "^GSPC"
elif "Dow Jones Industrial Average (^DJI)" in market_choice:
    market_ticker = "^DJI"
else:
    st.error(t("error_invalid_market_portfolio"))
    st.stop()

# 選択結果をSessionStateに保存
st.session_state.market_ticker = market_ticker


# ====計算ボタンと実行====

st.markdown("---")

# 計算ボタンを表示する条件
# CSVモードの場合: CSVファイルがアップロードされており，df_csvが存在する
# 銘柄検索モードの場合: close_dfが存在し，2銘柄以上が選択されている
show_calc_button = (
    (use_csv and "df_csv" in st.session_state and st.session_state.df_csv is not None and uploaded_file is not None) or
    (not use_csv and "close_df" in st.session_state and st.session_state.close_df is not None and len(st.session_state.selected_assets) >= 2)
)

# 計算ボタンの描画(押下時には run_calc がTrueになる)
if show_calc_button:
    run_calc = st.button(t("calc_button"), key="calc_button", disabled=st.session_state.calculating)

    # 計算開始時の準備(calculatingフラグを立て，画面上に「計算中です．」と表示)
    if run_calc:
        st.session_state.calculating = True
        with st.spinner(t("calculating")):
            use_csv = st.session_state.get("use_csv", False)
            if use_csv:  # CSV入力モード
                df = st.session_state.get("df_csv", None)
                if df is None:
                    st.error(t("error_csv_no_data"))
                    st.stop()
                # close_dfを定義（行＝日付，列＝銘柄）
                close_df = df.T
                try:
                    log_returns = calculate_log_returns(df, axis=1)  # df_csvからログリターン計算(axis=1で行方向に時系列計算)
                except Exception as e:
                    st.error(f"{t('error_log_return')}：{e}")
                    st.stop()
                # 証券コードリスト
                tickers = log_returns.index.tolist()
                log_returns = log_returns.T  # 計算後，log_returnsを転置して「行＝日付，列＝銘柄」に揃える
            else:  # 銘柄検索入力モード
                close_df = st.session_state.get("close_df", None)
                if close_df is None:
                    st.error(t("error_no_price_data"))
                    st.stop()
                try:
                    log_returns = calculate_log_returns(close_df, axis=0)  # close_dfからログリターン計算(axis=0で列方向に時系列計算)
                except Exception as e:
                    st.error(f"{t('error_log_return')}：{e}")
                    st.stop()
                # 有効な銘柄数チェック（2銘柄未満なら中断）
                if log_returns.shape[1] < 2:
                    st.error(t("error_valid_tickers"))
                    st.session_state.calculating = False
                    st.stop()
                # 日数のカウント（有効な共通データ行数）
                valid_days = log_returns.shape[0]
                # 想定されるスパン単位の日数
                expected_days = (end_date - start_date).days
                expected_count = {
                    t("daily"): expected_days,
                    t("weekly"): expected_days // 7,
                    t("monthly"): expected_days // 30
                }[span]
                # 警告表示
                if valid_days < expected_count * 1:
                    st.info(
                        f"{t('info_common_data')}{expected_count}{span}{t('info_common_data_only')}"
                        f"{valid_days}{span}{t('info_common_data_available')}"
                    )
                # 証券コードリスト
                tickers = log_returns.columns.tolist()
            
            st.session_state.close_df = close_df
            # log_returnsをセッションに保存
            st.session_state.log_returns = log_returns
            
            # 平均・標準偏差・共分散を計算（列＝銘柄方向）
            mean_returns = log_returns.mean(axis=0).values
            std_devs = log_returns.std(axis=0, ddof=0).values
            cov_matrix = np.cov(log_returns.T.values)
            
            # 最小投資割合のバリデーションチェック(全銘柄均等以上の最小割合は許容しない)
            N = len(tickers)
            if min_weight >= (1 / N):
                st.error(f"{t('error_min_weight_large')} {1/N:.4f}）．")
                st.session_state.calculating = False
                st.stop()
            if min_weight < 0 or min_weight >= 0.5:
                st.error(t("error_min_weight_range"))
                st.session_state.calculating = False
                st.stop()
            
            # 共分散行列に微小な正規化を加える(数値誤差防止)
            cov_matrix += np.eye(len(cov_matrix)) * 1e-10
            
            # 銘柄数
            N = len(mean_returns)
            
            # リターン(期待利益率)に関する統計量を取得
            sum_r = np.sum(mean_returns)  # リターンの合計
            max_r = np.max(mean_returns)  # 最大リターン
            min_r = np.min(mean_returns)  # 最小リターン
            
            # 最大リターンポートフォリオの期待利益率の計算
            max_weight = 1 - min_weight * (N - 1)
            max_return = max_r * max_weight + (sum_r - max_r) * min_weight
            
            # 最小リターンポートフォリオの期待利益率の計算
            min_return = min_r * max_weight + (sum_r - min_r) * min_weight
            
            # 数値誤差防止のための微小値を設定
            epsilon = 1e-6
            
            # 指定された段階数に応じてターゲットリターンを等間隔に設定
            target_returns = np.linspace(min_return + epsilon, max_return - epsilon, int(num_steps))
            
            # ポートフォリオのリスク(標準偏差)を計算する関数を定義
            def calculate_portfolio_volatility(weights, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 各ターゲットリターンに対する最小リスク点を求めるリスト
            frontier_vol = []
            frontier_weights = []
            
            # 最小分散フロンティアの各点を最適化で求める
            for target in target_returns:
                # 最適化条件
                constraints = (
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # ウェイトの合計は1
                    {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}  # 指定ターゲットリターンを達成
                )
                # 各銘柄のウェイトの上下限(最小投資割合以上)
                bounds = tuple((min_weight, 1.0) for _ in range(N))
                # 初期値は均等配分
                init_guess = np.array([1/N] * N)
                # 最適化実行(リスク最小化問題)
                result = minimize(
                    calculate_portfolio_volatility,
                    init_guess,
                    args=(cov_matrix,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-9}
                )
                # 成功した場合のみ結果を格納
                if result.success:
                    frontier_vol.append(result.fun)  # 最小リスク
                    frontier_weights.append(result.x)  # 最適なウェイト配分
            
            # 最適化結果が1点も得られなかった場合はエラー終了
            if len(frontier_vol) == 0:
                st.error(t("error_mvf_failed"))
                st.stop()

            # start_date,end_date,intervalの情報を取得
            start_date = st.session_state.get("start_date", None)
            end_date = st.session_state.get("end_date", None)
            interval = st.session_state.get("interval", None)
            if None in (start_date, end_date, interval):
                st.error(t("error_date_not_set"))
                st.stop()

            # yfinanceで市場ポートフォリオデータ取得
            market_ticker = st.session_state.get("market_ticker", None)
            try:
                market_data = yf.download(market_ticker, start=start_date, end=end_date, interval=interval, progress=False)
                # ダウンロードの失敗チェック
                if market_data is None or market_data.empty:
                    st.error(f"{t('error_market_data_failed')} {market_ticker}")
                    st.stop()
                # 市場リターンを計算
                market_returns = np.log(market_data["Close"] / market_data["Close"].shift(1)).dropna()
            except Exception as e:
                st.error(f"{t('error_market_data_fetch')}：{e}")
                st.stop()
            
            # β値の計算
            betas = {}
            market_returns_array = market_returns.squeeze()
            market_var = np.var(market_returns_array, ddof=0)
            for code in tickers:
                combined_df = pd.concat([log_returns[code], market_returns_array], axis=1, join='inner').dropna()
                if combined_df.shape[0] < 2:
                    beta = np.nan
                else:
                    cov = np.cov(combined_df.iloc[:, 0], combined_df.iloc[:, 1])[0, 1]
                    beta = cov / market_var if isinstance(market_var, (int, float)) and market_var != 0 else np.nan
                betas[code] = beta

            # 計算結果をセッションに保存
            st.session_state.result_data = {
                "tickers": tickers,
                "mean_returns": mean_returns,
                "std_devs": std_devs,
                "cov_matrix": cov_matrix,
                "target_returns": target_returns,
                "frontier_vol": frontier_vol,
                "frontier_weights": frontier_weights,
                "betas": betas,
                "market_return_mean": np.mean(market_returns),
                "risk_free_rate_span": rf_rate_span
            }
        
        st.session_state.calculating = False


# ====結果表示====

if st.session_state.result_data:
    st.markdown("---")
    st.markdown(f"## {t('analysis_results')}")
    
    data = st.session_state.result_data
    
    # 価格時系列の表示
    if not use_csv:
        if close_df is not None and not close_df.empty:
            with st.expander(t("price_time_series")):
                close_df_display = close_df.copy()
                close_df_display.index = close_df_display.index.strftime('%Y/%m/%d')  # 日付をYYYY/MM/DD形式に
                st.dataframe(close_df_display.round(2), use_container_width=True)
        else:
            st.error(t("error_no_price_data"))
            st.stop()
    
    # 為替レート（USD/JPY）の表示
    if st.session_state.get("convert_usd_to_jpy", False):
        try:
            fx_rates = fetch_usd_to_jpy_rates(start_date, end_date, interval)
            # 1列DataFrameの可能性があるのでSeries化
            if isinstance(fx_rates, pd.DataFrame):
                fx_rates = fx_rates["Close"] if "Close" in fx_rates.columns else fx_rates.squeeze()
            # 価格で使った日付（共通日付）に限定して為替レートを抽出
            common_dates = close_df.index
            fx_trimmed = fx_rates.reindex(common_dates).bfill().ffill()
            fx_df = fx_trimmed.rename("USD/JPY").to_frame()
            # 為替CSVデータをセッションに一時保存（DL後も再描画維持）
            st.session_state["fx_display_df"] = fx_df.to_csv(index=True).encode("utf-8")
            # タイトル
            with st.expander(t("fx_rate_display")):
                # Plotlyグラフ表示（Y軸の下限調整あり）
                fig_fx = go.Figure()
                fig_fx.add_trace(go.Scatter(
                    x=fx_trimmed.index,
                    y=fx_trimmed.values,
                    mode='lines+markers',
                    name='USD/JPY',
                    line=dict(color='lightblue'),
                    marker=dict(size=4)
                ))
                fig_fx.update_layout(
                    xaxis_title=t("date_label"),
                    yaxis_title=t("fx_rate_label"),
                    yaxis=dict(range=[fx_trimmed.min() * 0.995, fx_trimmed.max() * 1.005]),
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white', family='Meiryo'),
                    margin=dict(l=50, r=50, t=30, b=50),
                    height=400
                )
                st.plotly_chart(fig_fx, use_container_width=True)
                # CSVダウンロードボタン
                st.download_button(
                    label=t("fx_rate_download"),
                    data=st.session_state["fx_display_df"],
                    file_name="usd_jpy_rates.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.warning(f"{t('error_fx_display')}：{e}")
    
    # 銘柄ごとのリスク(log_returns標準偏差)・リターン(log_returns平均)情報を表示
    with st.expander(t("std_dev_return")):
        df_mean = pd.DataFrame({
            t("ticker_code"): data["tickers"],
            t("std_dev"): data["std_devs"],
            t("expected_return"): data["mean_returns"]
        })
        st.dataframe(df_mean.style.format({t("std_dev"): "{:.5f}", t("expected_return"): "{:.5f}"}), hide_index=True)
    
    # 相関行列の可視化（ヒートマップ）
    with st.expander(t("correlation_matrix")):
        # ログリターンをセッションから取得
        log_returns = st.session_state.get("log_returns", None)
        try:
            if "log_returns" in st.session_state:
                log_returns = st.session_state["log_returns"]
                tickers = log_returns.columns
                corr_matrix = log_returns.corr()
                st.session_state["corr_matrix"] = corr_matrix
            elif "corr_matrix" in st.session_state:
                corr_matrix = st.session_state["corr_matrix"]
                tickers = corr_matrix.columns
            else:
                st.warning(t("warning_correlation_data"))
                raise StopIteration
        
            corr_matrix = log_returns.corr()
            fig_corr, ax_corr = plt.subplots()
            cax = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax_corr.set_xticks(np.arange(len(tickers)))
            ax_corr.set_yticks(np.arange(len(tickers)))
            ax_corr.set_xticklabels(tickers, color='white', rotation=45, ha='right')
            ax_corr.set_yticklabels(tickers, color='white')
            ax_corr.tick_params(colors='white')
            cbar = fig_corr.colorbar(cax)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            fig_corr.patch.set_facecolor('black')
            ax_corr.set_facecolor('black')
            st.pyplot(fig_corr)

            # 相関係数のCSVダウンロードボタン
            st.session_state["corr_matrix_csv"] = corr_matrix.round(5).to_csv(index=True, encoding="utf-8-sig")
            st.download_button(
                label=t("correlation_download"),
                data=st.session_state["corr_matrix_csv"],
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"{t('error_correlation_display')}: {e}")

    # 最小分散フロンティアと資本市場線の表示
    with st.expander(t("mvf_cml_display")):
        if data["frontier_vol"] is None or len(data["frontier_vol"]) == 0:
            st.error(t("error_mvf_not_calculated"))
        else:
            # 最小分散点のインデックス
            min_index = np.nanargmin(data["frontier_vol"])
            efficient_vol = data["frontier_vol"][min_index:]
            efficient_returns = data["target_returns"][min_index:]
            
            # シャープレシオ最大点
            sharpe_ratios = (np.array(data["target_returns"]) - rf_rate_span) / np.array(data["frontier_vol"])
            max_sharpe_idx = np.nanargmax(sharpe_ratios)
            max_std = data["frontier_vol"][max_sharpe_idx]
            max_return = data["target_returns"][max_sharpe_idx]
            
            # CML描画用データ
            cml_x = np.linspace(0, max_std * 2, 100)
            cml_y = rf_rate_span + ((max_return - rf_rate_span) / max_std) * cml_x
            
            # Plotlyグラフ作成
            fig = go.Figure()
            
            # MVF（全体）
            fig.add_trace(go.Scatter(
                x=data["frontier_vol"],
                y=data["target_returns"],
                mode="lines",
                name="MVF",
                line=dict(color="gray", width=1), 
                zorder=3
            ))
            
            # 効率的フロンティア(EF)（MVFの右側）
            fig.add_trace(go.Scatter(
                x=efficient_vol,
                y=efficient_returns,
                mode="lines",
                name="EF",
                line=dict(color="cyan", width=2), 
                zorder=4
            ))
            
            # 最小分散ポートフォリオ
            fig.add_trace(go.Scatter(
                x=[data["frontier_vol"][min_index]],
                y=[data["target_returns"][min_index]],
                mode="markers",
                name="MVP",
                marker=dict(size=5, color="red", symbol="circle"), 
                zorder=5
            ))
            
            # CML
            fig.add_trace(go.Scatter(
                x=cml_x,
                y=cml_y,
                mode="lines",
                name="CML",
                line=dict(color="gold", width=2), 
                zorder=2
            ))
            
            # 無リスク利子率線
            fig.add_trace(go.Scatter(
                x=[0, max(data["frontier_vol"]) * 1.05],
                y=[rf_rate_span, rf_rate_span],
                mode="lines",
                name=f"RFR ({rf_rate_span:.3%})",
                line=dict(color="pink", dash="dot", width=1), 
                zorder=1
            ))
            
            # レイアウト設定
            fig.update_layout(
                xaxis=dict(title=t("std_dev"), showgrid=False),
                yaxis=dict(title=t("expected_return"), range=[min(rf_rate_span * 0.9, min(data["target_returns"]) * 0.9),
                                                            max(data["target_returns"]) * 1.1], showgrid=False),
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white", family='Meiryo'),
                legend=dict(x=1.02, y=1, borderwidth=0),
                margin=dict(r=150)
            )

            # 表示
            st.plotly_chart(fig, use_container_width=True)

            # 各期待利益率における標準偏差と投資割合をCSVとしてダウンロード
            weight_df = pd.DataFrame(data["frontier_weights"], columns=data["tickers"])
            weight_df.insert(0, t("std_dev"), data["frontier_vol"])
            weight_df.insert(1, t("expected_return"), data["target_returns"])
            weight_df = weight_df.sort_values(by=t("expected_return"), ascending=False)
            st.session_state["frontier_weights_csv"] = weight_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=t("mvf_cml_download"),
                data=st.session_state["frontier_weights_csv"],
                file_name="frontier_weights.csv",
                mime="text/csv"
            )
    
    # 証券市場線(SML)を表示
    with st.expander(t("sml_display")):
        # データ
        beta_vals = np.array(list(data["betas"].values()))
        expected_returns = np.array([data["mean_returns"][data["tickers"].index(code)] for code in data["betas"].keys()])
        rf = data["risk_free_rate_span"]
        rm = data["market_return_mean"]
        x_vals = np.linspace(0, max(2.5, beta_vals.max() * 1.2), 100)
        sml_y = rf + (rm - rf) * x_vals

        # SMLライン
        sml_line = go.Scatter(
            x=x_vals,
            y=sml_y,
            mode='lines',
            name=f'SML ({st.session_state.market_ticker})',
            line=dict(width=1, color='gold')
        )

        # 各銘柄の点
        tickers_label = t("tickers") if st.session_state.language == "ja" else "Tickers"
        stock_points = go.Scatter(
            x=beta_vals,
            y=expected_returns,
            mode='markers+text',
            name=tickers_label,
            text=list(data["betas"].keys()),
            textposition="top center",
            textfont=dict(size=10, color='lightgray'),
            marker=dict(size=5, color='lightblue')
        )

        # 無リスク利子率の横線
        rf_line = go.Scatter(
            x=[0, max(x_vals)],
            y=[rf, rf],
            mode='lines',
            name=f'RFR ({rf:.3%})',
            line=dict(color="pink", dash="dot", width=1)
        )

        # レイアウト
        layout = go.Layout(
            xaxis=dict(title=t("beta"), showgrid=False),
            yaxis=dict(title=t("expected_return"), showgrid=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white', family='Meiryo'),
            legend=dict(x=1.05, y=1, borderwidth=0)
        )
        
        fig = go.Figure(data=[sml_line, stock_points, rf_line], layout=layout)
        
        # 表示
        st.plotly_chart(fig, use_container_width=True)

        # 各銘柄のβ値をCSVとしてダウンロード
        beta_df = pd.DataFrame({
            t("ticker_code"): list(data["betas"].keys()),
            t("beta_value"): list(data["betas"].values())
        })
        st.session_state["beta_csv"] = beta_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=t("beta_download"),
            data=st.session_state["beta_csv"],
            file_name="beta.csv",
            mime="text/csv"
        )


# コメント
st.markdown(f"""
    <hr style="margin-top: 3rem; margin-bottom: 1rem; border: none; border-top: 1px solid #444;">
    <div style='text-align: left; font-size: 0.8rem; color: gray;'>
        {t("disclaimer")}
    </div>
""", unsafe_allow_html=True)
