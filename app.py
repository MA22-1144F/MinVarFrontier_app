"""
最小分散フロンティア計算アプリケーション
Minimum Variance Frontier Calculator
"""

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
# 時間操作のモジュール
import time
# CSVテンプレート生成などに使用
import io
# 並列実行用
import concurrent.futures
# メモリ上にテキストストリームを作成
from io import StringIO
# 数学・統計処理(最小分散フロンティアの計算)
from scipy.optimize import minimize
# 日付操作(前営業日の取得など)
from datetime import date, timedelta
# プロット用ライブラリ
import matplotlib.pyplot as plt
# グラフ描画用ライブラリ
import plotly.graph_objs as go

# ====言語設定====
# 言語辞書
LANG = {
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
        "results_count": " results",
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
        "jgb_rate_info": "Short-term JGB rate on",
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
        "disclaimer": "This app is for educational purposes only and not intended for investment decisions.<br>The developer assumes no responsibility for any damages arising from use of this app.",
        "error_future_date": "End date is in the future. Please select a valid end date.",
        "error_date_order": "Start date must be earlier than end date.",
        "error_min_tickers": "You need to add at least 2 tickers to the list.",
        "error_price_fetch": "Price data fetch error",
        "error_price_empty": "Failed to fetch price data.",
        "error_csv_date_format": "Invalid date format.",
        "error_csv_processing": "Error occurred during CSV data processing",
        "error_log_return": "Failed to calculate log returns",
        "error_valid_tickers": "At least 2 tickers with valid data are required. Please review the period.",
        "info_common_data": "For the specified period ({expected} {span}), only {valid} {span} of common valid price data is available.",
        "error_min_weight_large": "Minimum investment ratio is too large for the selected number of tickers (must be less than {limit:.4f}).",
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
        "error_invalid_market_portfolio": "Invalid market portfolio selected.",
        "date_label": "Date",
        "exchange_rate": "Exchange Rate",
        "language_select": "Language",
        "warning_jp_stock_list_failed": "Failed to retrieve Japanese stock list",
        "warning_search_error": "Search error",
        "warning_data_fetch_error": "Data fetch error for",
        "warning_fx_conversion_failed": "Failed to convert currency",
        "error_no_valid_price_data": "Failed to retrieve any valid price data.",
        "error_no_valid_tickers": "No valid tickers provided.",
        "mof_source": "Ministry of Finance",
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
        "disclaimer": "本アプリは学習目的で作成されたものであり，投資判断への利用を想定したものではありません．<br>本アプリの利用によって生じたいかなる損害についても開発者は責任を負いかねます．",
        "error_future_date": "終了日が未来の日付になっています．正しい終了日を選んでください．",
        "error_date_order": "開始日は終了日より前の日付を選択してください．",
        "error_min_tickers": "2銘柄以上をリストに追加する必要があります．",
        "error_price_fetch": "価格データ取得エラー",
        "error_price_empty": "価格データの取得に失敗しました．",
        "error_csv_date_format": "日付の形式が不正です．",
        "error_csv_processing": "CSVのデータ処理中にエラーが発生しました",
        "error_log_return": "ログリターンの計算に失敗しました",
        "error_valid_tickers": "有効なデータを持つ銘柄が2つ以上必要です．期間を見直してください．",
        "info_common_data": "指定された期間（{expected}{span}）に対し，共通の有効価格データが存在するのは{valid}{span}のみです．",
        "error_min_weight_large": "選択された銘柄数に対して最小投資割合が大きすぎます（{limit:.4f}未満である必要があります）．",
        "error_min_weight_range": "最小投資割合は0以上0.5未満である必要があります．",
        "error_mvf_failed": "最小分散フロンティアの計算に失敗しました．銘柄数・期間・最小投資割合を見直してください．",
        "error_date_not_set": "日付が正しく設定されていません．",
        "error_market_data_failed": "市場ポートフォリオのデータ取得に失敗しました：",
        "error_market_data_fetch": "市場データ取得中にエラーが発生しました",
        "error_no_price_data": "価格データが存在しません．",
        "error_fx_display": "為替レートの表示に失敗しました",
        "error_correlation_display": "相関行列の表示中にエラーが発生しました",
        "error_mvf_not_calculated": "最小分散フロンティアが正常に計算できませんでした．",
        "warning_correlation_data": "相関行列を表示するには先に価格データの取得が必要です．",
        "error_csv_no_data": "CSVファイルのデータが見つかりません．",
        "error_span_estimation": "スパンの推定中にエラーが発生しました．",
        "error_invalid_market_portfolio": "無効な市場ポートフォリオが選択されました．",
        "date_label": "日付",
        "exchange_rate": "為替レート",
        "language_select": "言語",
        "warning_jp_stock_list_failed": "日本銘柄リストの取得に失敗しました",
        "warning_search_error": "検索エラー",
        "warning_data_fetch_error": "のデータ取得エラー",
        "warning_fx_conversion_failed": "為替換算に失敗しました",
        "error_no_valid_price_data": "有効な価格データが一つも取得できませんでした．",
        "error_no_valid_tickers": "有効なティッカーが1つもありません．",
        "mof_source": "財務省",
    }
}

# ====ページ設定====
st.set_page_config(page_title="MVF Calculator", layout="wide")

st.markdown("""
<style>
.block-container {
    background-color: #000;
    color: #fff;
    font-family: Meiryo, sans-serif;
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}
div.stButton > button:first-child {
    width: 100% !important;
    text-align: center !important;
}
header, footer, .stActionButton, .stDeployButton {
    display: none !important;
}
.custom-title {
    text-align: center;
    font-size: 40px;
    margin-bottom: 1rem;
}
@media screen and (max-width: 600px) {
    .custom-title { font-size: 25px; }
}
</style>
""", unsafe_allow_html=True)

# ====セッションステート初期化====
DEFAULTS = {
    'language': 'en',
    'calculating': False,
    'result_data': None,
    'selected_assets': [],
    'previous_input_mode': None,
    'convert_usd_to_jpy': False,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ====ユーティリティ関数====
def t(key):
    """言語辞書からテキスト取得"""
    return LANG[st.session_state.language].get(key, key)

def normalize_input(text):
    """全角→半角変換"""
    return unicodedata.normalize('NFKC', text)

def format_date(dt):
    """日付をYYYY/MM/DD形式に変換"""
    return dt.strftime('%Y/%m/%d') if hasattr(dt, 'strftime') else str(dt)

# ====データ取得関数====
def get_jgb_rate():
    """財務省から短期国債金利取得"""
    try:
        url = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        for enc in ['shift_jis', 'utf-8', 'cp932']:
            try:
                df = pd.read_csv(StringIO(resp.content.decode(enc)), header=None)
                break
            except:
                continue
        else:
            return None
        
        valid = df[df.iloc[:, 1].notna()]
        if valid.empty:
            return None
        
        idx = valid.index[-1]
        return url, df.iloc[idx, 0], float(df.iloc[idx, 1])
    except:
        return None

@st.cache_data
def load_japan_stocks():
    """日本株リスト取得"""
    try:
        url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        df = pd.read_excel(url, skiprows=1, header=None)[[1, 2]].dropna()
        df.columns = ['code', 'name']
        df['code'] = df['code'].astype(str).str.strip()
        return df
    except Exception as e:
        st.warning(f"{t('warning_jp_stock_list_failed')}: {e}")
        return None

@st.cache_data
def fetch_fx_rates(start, end, interval):
    """ドル円レート取得"""
    ticker = yf.Ticker("JPY=X")
    hist = ticker.history(
        start=start,
        end=end + timedelta(days=1),
        interval=interval,
        auto_adjust=False,
        prepost=False,
        repair=True
    )
    
    if hist is None or hist.empty:
        raise ValueError("FX rate fetch failed")
    
    # Close列を取得
    if 'Adj Close' in hist.columns:
        fx = hist['Adj Close'].copy()
    elif 'Close' in hist.columns:
        fx = hist['Close'].copy()
    else:
        raise ValueError("FX rate fetch failed - no Close column")
    
    # タイムゾーン処理
    if hasattr(fx.index, 'tz') and fx.index.tz is not None:
        fx.index = fx.index.tz_localize(None)
    
    # 日付正規化
    fx.index = fx.index.normalize()
    
    return fx.dropna().sort_index()

def fetch_single_asset(args):
    """単一銘柄データ取得"""
    symbol, start, end, interval = args
    try:
        # yfinanceでデータ取得
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start,
            end=end + timedelta(days=1),  # 終了日を含むため+1日
            interval=interval,
            auto_adjust=False,  # Adj Close取得に必須
            prepost=False,      # プレ・ポストマーケットを除外
            repair=True         # データの修復を有効化
        )
        
        if not hist.empty and 'Adj Close' in hist.columns:
            # 調整後終値を取得
            price_series = hist['Adj Close'].copy()
            price_series.name = symbol
            
            # 日付データはyfinanceの元データをそのまま使用
            # タイムゾーン情報がある場合のみ削除
            if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
                price_series.index = price_series.index.tz_localize(None)
            
            # 日付のみに正規化（時間部分を削除）
            price_series.index = price_series.index.normalize()
            
            # NaN値を除去（補完は行わない）
            price_series = price_series.dropna()
            
            if not price_series.empty:
                return {
                    'symbol': symbol,
                    'data': price_series,
                    'success': True
                }
        
        # Adj Closeがない場合はCloseを使用
        if not hist.empty and 'Close' in hist.columns:
            price_series = hist['Close'].copy()
            price_series.name = symbol
            
            if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
                price_series.index = price_series.index.tz_localize(None)
            
            price_series.index = price_series.index.normalize()
            price_series = price_series.dropna()
            
            if not price_series.empty:
                return {
                    'symbol': symbol,
                    'data': price_series,
                    'success': True
                }
        
        return {
            'symbol': symbol,
            'data': pd.Series(dtype=float),
            'success': False
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'data': pd.Series(dtype=float),
            'success': False,
            'error': str(e)
        }

def fetch_prices(symbols, start, end, interval):
    """複数銘柄の価格データ並列取得"""
    if not symbols:
        raise ValueError(t("error_no_valid_tickers"))
    
    args_list = [(s, start, end, interval) for s in symbols]
    data = {}
    failed_symbols = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(symbols))) as ex:
        futures = {ex.submit(fetch_single_asset, a): a[0] for a in args_list}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r['success'] and not r['data'].empty:
                data[r['symbol']] = r['data']
            else:
                failed_symbols.append(r['symbol'])
    
    # 失敗した銘柄を警告
    if failed_symbols:
        st.warning(f"{t('warning_data_fetch_error')}: {', '.join(failed_symbols)}")
    
    if not data:
        raise ValueError(t("error_no_valid_price_data"))
    
    df = pd.DataFrame(data).sort_index().dropna()
    
    # 為替換算
    if st.session_state.get("convert_usd_to_jpy"):
        try:
            fx = fetch_fx_rates(start, end, interval)
            if isinstance(fx, pd.DataFrame):
                fx = fx.squeeze()
            for sym in df.columns:
                if not sym.endswith(".T"):
                    aligned = fx.reindex(df.index).ffill().bfill()
                    if len(aligned) == len(df.index):
                        df[sym] = df[sym] * aligned
        except Exception as e:
            st.warning(f"{t('warning_fx_conversion_failed')}: {e}")
    
    return df

def calc_log_returns(df, axis="auto"):
    """ログリターン計算"""
    if df.isnull().values.any():
        raise ValueError("Missing values in data")
    if (df <= 0).values.any():
        raise ValueError("Non-positive values in data")
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 columns")
    
    df = df.sort_index()
    
    if axis == "auto":
        if pd.api.types.is_datetime64_any_dtype(df.columns):
            axis = 1
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            axis = 0
        else:
            raise ValueError("Cannot detect time series axis")
    
    return np.log(df / df.shift(axis=axis)).dropna(axis=axis)

# ====銘柄検索クラス====
class AssetSearcher:
    def __init__(self, jp_df=None):
        self.url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        self.last_req = 0
        self.jp_df = jp_df
    
    def search(self, query, max_results=20):
        if not query or len(query.strip()) < 1:
            return []
        
        query = query.strip()
        results = []
        
        if self.jp_df is not None:
            results.extend(self._search_jp(query))
        
        try:
            self._rate_limit()
            for r in self._call_api(query):
                asset = self._convert(r)
                if asset and not any(a['symbol'] == asset['symbol'] for a in results):
                    results.append(asset)
        except Exception as e:
            st.warning(f"{t('warning_search_error')}: {e}")
        
        return results[:max_results]
    
    def _search_jp(self, query):
        results = []
        try:
            code_match = self.jp_df[self.jp_df['code'].str.contains(query, case=False, na=False)]
            name_match = self.jp_df[self.jp_df['name'].str.contains(query, case=False, na=False)]
            matches = pd.concat([code_match, name_match]).drop_duplicates().head(20)
            for _, row in matches.iterrows():
                results.append({
                    'symbol': row['code'] + '.T',
                    'name': row['name'],
                    'exchange': 'Tokyo',
                    'currency': 'JPY',
                    'type': 'EQUITY'
                })
        except:
            pass
        return results
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_req
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self.last_req = time.time()
    
    def _call_api(self, query):
        try:
            resp = self.session.get(self.url, params={'q': query, 'quotesCount': 15, 'newsCount': 0}, timeout=3)
            resp.raise_for_status()
            return resp.json().get('quotes', [])
        except:
            return []
    
    def _convert(self, r):
        symbol = r.get('symbol')
        if not symbol:
            return None
        return {
            'symbol': symbol,
            'name': r.get('longname') or r.get('shortname') or symbol,
            'exchange': r.get('exchange', ''),
            'currency': r.get('currency', 'USD'),
            'type': r.get('quoteType', '')
        }

# ====メインUI====

# 言語選択
col1, col2 = st.columns([4, 1])
with col2:
    lang_opts = {"English": "en", "日本語": "ja"}
    selected = st.selectbox(
        t("language_select"),
        list(lang_opts.keys()),
        index=0 if st.session_state.language == "en" else 1
    )
    if lang_opts[selected] != st.session_state.language:
        st.session_state.language = lang_opts[selected]
        st.rerun()

# タイトル
st.markdown(f"<div class='custom-title'>{t('app_title')}</div>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid white; margin: 25px 0;'/>", unsafe_allow_html=True)

# 入力モード選択
input_mode = st.radio(t("data_input_method"), [t("ticker_search_input"), t("csv_input")])
use_csv = input_mode == t("csv_input")

# モード変更時の初期化
if st.session_state.previous_input_mode != input_mode:
    for key in ['df_csv', 'close_df', 'log_returns', 'result_data']:
        st.session_state[key] = None
    st.session_state.calculating = False
    st.session_state.previous_input_mode = input_mode

st.session_state.use_csv = use_csv
st.markdown("---")

# ====CSV入力モード====
if use_csv:
    uploaded = st.file_uploader(t("csv_file_upload"), type="csv")
    
    # サンプルCSV
    sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=10, freq="B")
    sample_data = pd.DataFrame({
        d.strftime("%Y-%m-%d"): [
            np.random.randint(2500, 2750),
            np.random.randint(3250, 3750),
            np.random.randint(140, 150)
        ] for d in sample_dates
    }, index=["7203.T", "6758.T", "9432.T"])
    sample_data.index.name = ""
    
    csv_buf = io.StringIO()
    sample_data.to_csv(csv_buf)
    st.download_button(t("sample_csv_download"), csv_buf.getvalue(), "sample.csv", "text/csv")
    
    if uploaded:
        try:
            df_csv = pd.read_csv(uploaded, index_col=0)
            dates = pd.to_datetime(df_csv.columns, errors='coerce')
            if dates.isnull().any():
                raise ValueError(t("error_csv_date_format"))
            
            df_csv.columns = dates
            df_csv = df_csv.sort_index(axis=1)
            
            st.session_state.start_date = df_csv.columns.min().date()
            st.session_state.end_date = df_csv.columns.max().date()
            
            # スパン推定
            deltas = np.diff(df_csv.columns).astype('timedelta64[D]').astype(int)
            avg = np.mean(deltas)
            if avg <= 2:
                span, interval = t("daily"), "1d"
            elif avg <= 10:
                span, interval = t("weekly"), "1wk"
            else:
                span, interval = t("monthly"), "1mo"
            
            st.session_state.interval = interval
            st.session_state.span = span
            st.info(f"{t('span_auto_detected')}: **{span}** ({t('avg_interval')} {avg:.1f} {t('days')})")
            
            with st.expander(t("csv_preview")):
                display = df_csv.copy()
                display.columns = [format_date(c) for c in display.columns]
                st.dataframe(display)
            
            st.markdown(f"<p style='font-size: 16px; color: lightgray;'>{t('analysis_target_count')}: <strong>{df_csv.shape[0]}</strong> {t('tickers')}</p>", unsafe_allow_html=True)
            st.session_state.df_csv = df_csv
            
        except Exception as e:
            st.error(f"{t('error_csv_processing')}: {e}")

# ====銘柄検索入力モード====
else:
    st.markdown(f"### {t('ticker_search')}")
    
    if 'jp_stocks' not in st.session_state:
        st.session_state.jp_stocks = load_japan_stocks()
    
    if 'searcher' not in st.session_state:
        st.session_state.searcher = AssetSearcher(st.session_state.jp_stocks)
    
    search_query = normalize_input(st.text_input(
        t("ticker_search_label"),
        placeholder=t("ticker_search_placeholder")
    ))
    
    st.checkbox(t("convert_usd_to_jpy"), key="convert_usd_to_jpy")
    
    # 検索結果表示
    if search_query:
        with st.spinner(t("calculating")):
            results = st.session_state.searcher.search(search_query)
        
        if results:
            st.markdown(f"**{t('search_results')}: {len(results)}{t('results_count')}**")
            
            for i, asset in enumerate(results):
                c1, c2, c3, c4 = st.columns([2, 4, 2, 1])
                c1.write(f"**{asset['symbol']}**")
                c2.write(asset['name'])
                c3.write(f"{asset['currency']} ({asset['exchange'] or 'N/A'})")
                
                is_sel = any(a['symbol'] == asset['symbol'] for a in st.session_state.selected_assets)
                if is_sel:
                    c4.write(t("already_selected"))
                elif c4.button(t("add_button"), key=f"add_{i}_{asset['symbol']}"):
                    st.session_state.selected_assets.append(asset)
                    st.rerun()
        else:
            st.warning(t("no_results"))
    
    # 選択済み銘柄
    if st.session_state.selected_assets:
        st.markdown("---")
        st.markdown(f"### {t('selected_tickers')}")
        st.markdown(f"<p style='font-size: 16px; color: lightgray;'>{t('analysis_target_count')}: <strong>{len(st.session_state.selected_assets)}</strong> {t('tickers')}</p>", unsafe_allow_html=True)
        
        for i, asset in enumerate(st.session_state.selected_assets):
            cols = st.columns([2, 4, 2, 1])
            cols[0].write(asset["symbol"])
            cols[1].write(asset["name"])
            cols[2].write(f"{asset['currency']} ({asset.get('exchange', 'N/A')})")
            if cols[3].button(t("delete_button"), key=f"del_{i}"):
                st.session_state.selected_assets.pop(i)
                st.rerun()
        
        if st.button(t("reset_button"), type="secondary"):
            st.session_state.selected_assets = []
            st.session_state.result_data = None
            st.rerun()
        
        # 期間設定
        st.markdown("---")
        st.markdown(f"### {t('analysis_period')}")
        
        def_end = date.today() - timedelta(days=1)
        def_start = def_end - timedelta(days=365)
        start_date = st.date_input(t("start_date"), def_start)
        end_date = st.date_input(t("end_date"), def_end)
        
        if end_date > date.today():
            st.error(t("error_future_date"))
            st.stop()
        if start_date >= end_date:
            st.error(t("error_date_order"))
            st.stop()
        
        span = st.radio(t("span_label"), [t("daily"), t("weekly"), t("monthly")])
        interval_map = {t("daily"): "1d", t("weekly"): "1wk", t("monthly"): "1mo"}
        interval = interval_map[span]
        
        symbols = [a["symbol"] for a in st.session_state.selected_assets]
        
        if len(symbols) < 2:
            st.info(t("error_min_tickers"))
            st.stop()
        
        try:
            with st.spinner(t("calculating")):
                close_df = fetch_prices(symbols, start_date, end_date, interval)
        except ValueError as e:
            st.error(f"{t('error_price_fetch')}: {e}")
            st.stop()
        
        if close_df.empty:
            st.error(t("error_price_empty"))
            st.stop()
        
        st.session_state.close_df = close_df.sort_index()
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.interval = interval
        st.session_state.span = span

# ====分析パラメータ====
st.markdown("---")
st.markdown(f"### {t('analysis_params')}")

min_weight = st.number_input(t("min_weight"), 0.0, 0.5, 0.0, 0.001, "%.3f")
num_steps = st.number_input(t("num_steps"), 5, 500, 50)

# リスクフリーレート
jgb = get_jgb_rate()
if jgb:
    url, dt, rate = jgb
    if st.session_state.language == "ja":
        st.info(f"{dt}{t('jgb_rate_info')}{rate:.3f}%です．[{t('mof_source')}]({url})")
    else:
        st.info(f"{t('jgb_rate_info')} {dt}: {rate:.3f}% [{t('mof_source')}]({url})")
    rf_default = rate
else:
    st.warning(t("jgb_rate_failed"))
    rf_default = 0.5

rf_rate = st.number_input(t("risk_free_rate_label"), 0.0, 100.0, rf_default, 0.001, "%.3f") / 100

# ====比較市場ポートフォリオの選択====
st.markdown("---")
st.markdown(f"### {t('market_portfolio')}")

MARKETS = {
    "Nikkei 225 (^N225)": "^N225",
    "NASDAQ Composite (^IXIC)": "^IXIC",
    "S&P 500 (^GSPC)": "^GSPC",
    "Dow Jones Industrial Average (^DJI)": "^DJI"
}

market_choice = st.radio(t("market_portfolio_label"), list(MARKETS.keys()))
market_ticker = MARKETS.get(market_choice)
if not market_ticker:
    st.error(t("error_invalid_market_portfolio"))
    st.stop()

st.session_state.market_ticker = market_ticker

# ====計算実行====
st.markdown("---")

can_calc = (
    (use_csv and st.session_state.get("df_csv") is not None) or
    (not use_csv and st.session_state.get("close_df") is not None and len(st.session_state.selected_assets) >= 2)
)

if can_calc and st.button(t("calc_button"), disabled=st.session_state.calculating):
    st.session_state.calculating = True
    
    with st.spinner(t("calculating")):
        try:
            # データ準備
            if use_csv:
                df = st.session_state.df_csv
                if df is None:
                    raise ValueError(t("error_csv_no_data"))
                close_df = df.T
                log_returns = calc_log_returns(df, axis=1).T
                tickers = log_returns.columns.tolist()
            else:
                close_df = st.session_state.close_df
                if close_df is None:
                    raise ValueError(t("error_no_price_data"))
                log_returns = calc_log_returns(close_df, axis=0)
                tickers = log_returns.columns.tolist()
            
            if log_returns.shape[1] < 2:
                raise ValueError(t("error_valid_tickers"))
            
            st.session_state.close_df = close_df
            st.session_state.log_returns = log_returns
            
            # 統計量計算
            mean_ret = log_returns.mean().values
            std_devs = log_returns.std(ddof=0).values
            cov_mat = np.cov(log_returns.T.values) + np.eye(len(tickers)) * 1e-10
            
            N = len(tickers)
            
            # 最小投資割合チェック
            if min_weight >= 1 / N:
                raise ValueError(t("error_min_weight_large").format(limit=1/N))
            if min_weight < 0 or min_weight >= 0.5:
                raise ValueError(t("error_min_weight_range"))
            
            # リスクフリーレートをスパンに変換
            span = st.session_state.get("span", t("daily"))
            if span == t("daily"):
                rf_span = rf_rate / 252
            elif span == t("weekly"):
                rf_span = rf_rate / 52
            else:
                rf_span = rf_rate / 12
            
            # ターゲットリターン設定
            max_w = 1 - min_weight * (N - 1)
            max_r = max(mean_ret) * max_w + (sum(mean_ret) - max(mean_ret)) * min_weight
            min_r = min(mean_ret) * max_w + (sum(mean_ret) - min(mean_ret)) * min_weight
            eps = 1e-6
            targets = np.linspace(min_r + eps, max_r - eps, int(num_steps))
            
            # 最小分散フロンティア計算
            def port_vol(w, cov):
                return np.sqrt(w.T @ cov @ w)
            
            frontier_vol, frontier_w = [], []
            
            for tgt in targets:
                cons = (
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'eq', 'fun': lambda w, t=tgt: w @ mean_ret - t}
                )
                bounds = [(min_weight, 1.0)] * N
                res = minimize(port_vol, [1/N]*N, args=(cov_mat,), method='SLSQP',
                             bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
                if res.success:
                    frontier_vol.append(res.fun)
                    frontier_w.append(res.x)
            
            if not frontier_vol:
                raise ValueError(t("error_mvf_failed"))
            
            # 市場データ取得・β計算
            start = st.session_state.get("start_date")
            end = st.session_state.get("end_date")
            interval = st.session_state.get("interval")
            
            # 市場ポートフォリオデータ取得
            mkt_ticker = yf.Ticker(market_ticker)
            mkt_hist = mkt_ticker.history(
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                prepost=False,
                repair=True
            )
            
            if mkt_hist is None or mkt_hist.empty:
                raise ValueError(f"{t('error_market_data_failed')} {market_ticker}")
            
            # Adj CloseまたはCloseを取得
            if 'Adj Close' in mkt_hist.columns:
                mkt_close = mkt_hist['Adj Close'].copy()
            elif 'Close' in mkt_hist.columns:
                mkt_close = mkt_hist['Close'].copy()
            else:
                raise ValueError(f"{t('error_market_data_failed')} {market_ticker}")
            
            # タイムゾーン処理
            if hasattr(mkt_close.index, 'tz') and mkt_close.index.tz is not None:
                mkt_close.index = mkt_close.index.tz_localize(None)
            
            # 日付正規化
            mkt_close.index = mkt_close.index.normalize()
            mkt_close = mkt_close.dropna()
            
            mkt_ret = np.log(mkt_close / mkt_close.shift(1)).dropna()
            mkt_var = np.var(mkt_ret, ddof=0)
            
            betas = {}
            for code in tickers:
                combo = pd.concat([log_returns[code], mkt_ret], axis=1, join='inner').dropna()
                if combo.shape[0] < 2:
                    betas[code] = np.nan
                else:
                    cov = np.cov(combo.iloc[:, 0], combo.iloc[:, 1])[0, 1]
                    betas[code] = cov / mkt_var if mkt_var != 0 else np.nan
            
            st.session_state.result_data = {
                "tickers": tickers,
                "mean_returns": mean_ret,
                "std_devs": std_devs,
                "cov_matrix": cov_mat,
                "target_returns": targets,
                "frontier_vol": frontier_vol,
                "frontier_weights": frontier_w,
                "betas": betas,
                "market_return_mean": np.mean(mkt_ret),
                "risk_free_rate_span": rf_span
            }
            
        except Exception as e:
            st.error(str(e))
    
    st.session_state.calculating = False

# ====結果表示====
if st.session_state.result_data:
    st.markdown("---")
    st.markdown(f"## {t('analysis_results')}")
    
    data = st.session_state.result_data
    close_df = st.session_state.get("close_df")
    log_returns = st.session_state.get("log_returns")
    rf_span = data["risk_free_rate_span"]
    
    # 価格時系列
    if not use_csv and close_df is not None and not close_df.empty:
        with st.expander(t("price_time_series")):
            disp = close_df.copy()
            disp.index = [format_date(d) for d in disp.index]
            st.dataframe(disp.round(2), use_container_width=True)
    
    # 為替レート
    if st.session_state.get("convert_usd_to_jpy") and close_df is not None:
        try:
            fx = fetch_fx_rates(st.session_state.start_date, st.session_state.end_date, st.session_state.interval)
            if isinstance(fx, pd.DataFrame):
                fx = fx.squeeze()
            fx = fx.reindex(close_df.index).bfill().ffill()
            
            with st.expander(t("fx_rate_display")):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fx.index, y=fx.values, mode='lines+markers',
                                        name='USD/JPY', line=dict(color='lightblue'), marker=dict(size=4)))
                fig.update_layout(
                    xaxis_title=t("date_label"), yaxis_title=t("fx_rate_label"),
                    yaxis=dict(range=[fx.min() * 0.995, fx.max() * 1.005]),
                    plot_bgcolor='black', paper_bgcolor='black',
                    font=dict(color='white', family='Meiryo'), height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(t("fx_rate_download"), fx.to_csv().encode("utf-8"), "usd_jpy_rates.csv", "text/csv")
        except Exception as e:
            st.warning(f"{t('error_fx_display')}: {e}")
    
    # 標準偏差・期待リターン
    with st.expander(t("std_dev_return")):
        df_stats = pd.DataFrame({
            t("ticker_code"): data["tickers"],
            t("std_dev"): data["std_devs"],
            t("expected_return"): data["mean_returns"]
        })
        st.dataframe(df_stats.style.format({t("std_dev"): "{:.5f}", t("expected_return"): "{:.5f}"}), hide_index=True)
    
    # 相関行列
    if log_returns is not None:
        with st.expander(t("correlation_matrix")):
            try:
                corr = log_returns.corr()
                fig, ax = plt.subplots()
                cax = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, color='white', rotation=45, ha='right')
                ax.set_yticklabels(corr.columns, color='white')
                ax.tick_params(colors='white')
                cbar = fig.colorbar(cax)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                st.pyplot(fig)
                st.download_button(t("correlation_download"), corr.round(5).to_csv().encode("utf-8-sig"), "correlation_matrix.csv", "text/csv")
            except Exception as e:
                st.error(f"{t('error_correlation_display')}: {e}")
    
    # MVF & CML
    with st.expander(t("mvf_cml_display")):
        if not data["frontier_vol"]:
            st.error(t("error_mvf_not_calculated"))
        else:
            fv = data["frontier_vol"]
            tr = data["target_returns"]
            
            min_idx = np.nanargmin(fv)
            eff_vol = fv[min_idx:]
            eff_ret = tr[min_idx:]
            
            sharpe = (np.array(tr) - rf_span) / np.array(fv)
            max_sh_idx = np.nanargmax(sharpe)
            max_std, max_ret = fv[max_sh_idx], tr[max_sh_idx]
            
            cml_x = np.linspace(0, max_std * 2, 100)
            cml_y = rf_span + ((max_ret - rf_span) / max_std) * cml_x
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fv, y=tr, mode="lines", name="MVF", line=dict(color="gray", width=1)))
            fig.add_trace(go.Scatter(x=eff_vol, y=eff_ret, mode="lines", name="EF", line=dict(color="cyan", width=2)))
            fig.add_trace(go.Scatter(x=[fv[min_idx]], y=[tr[min_idx]], mode="markers", name="MVP", marker=dict(size=5, color="red")))
            fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode="lines", name="CML", line=dict(color="gold", width=2)))
            fig.add_trace(go.Scatter(x=[0, max(fv) * 1.05], y=[rf_span, rf_span], mode="lines", name=f"RFR ({rf_span:.3%})", line=dict(color="pink", dash="dot", width=1)))
            
            fig.update_layout(
                xaxis=dict(title=t("std_dev"), showgrid=False),
                yaxis=dict(title=t("expected_return"), range=[min(rf_span * 0.9, min(tr) * 0.9), max(tr) * 1.1], showgrid=False),
                plot_bgcolor="black", paper_bgcolor="black",
                font=dict(color="white", family='Meiryo'),
                legend=dict(x=1.02, y=1, borderwidth=0), margin=dict(r=150)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            w_df = pd.DataFrame(data["frontier_weights"], columns=data["tickers"])
            w_df.insert(0, t("std_dev"), fv)
            w_df.insert(1, t("expected_return"), tr)
            w_df = w_df.sort_values(by=t("expected_return"), ascending=False)
            st.download_button(t("mvf_cml_download"), w_df.to_csv(index=False).encode("utf-8-sig"), "frontier_weights.csv", "text/csv")
    
    # SML
    with st.expander(t("sml_display")):
        betas = data["betas"]
        beta_vals = np.array(list(betas.values()))
        exp_ret = np.array([data["mean_returns"][data["tickers"].index(c)] for c in betas.keys()])
        rm = data["market_return_mean"]
        
        x_vals = np.linspace(0, max(2.5, beta_vals.max() * 1.2), 100)
        sml_y = rf_span + (rm - rf_span) * x_vals
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=sml_y, mode='lines', name=f'SML ({market_ticker})', line=dict(width=1, color='gold')))
        fig.add_trace(go.Scatter(x=beta_vals, y=exp_ret, mode='markers+text', name=t("tickers"),
                                text=list(betas.keys()), textposition="top center",
                                textfont=dict(size=10, color='lightgray'), marker=dict(size=5, color='lightblue')))
        fig.add_trace(go.Scatter(x=[0, max(x_vals)], y=[rf_span, rf_span], mode='lines',
                                name=f'RFR ({rf_span:.3%})', line=dict(color="pink", dash="dot", width=1)))
        
        fig.update_layout(
            xaxis=dict(title=t("beta"), showgrid=False),
            yaxis=dict(title=t("expected_return"), showgrid=False),
            plot_bgcolor='black', paper_bgcolor='black',
            font=dict(color='white', family='Meiryo'),
            legend=dict(x=1.05, y=1, borderwidth=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        beta_df = pd.DataFrame({t("ticker_code"): list(betas.keys()), t("beta_value"): list(betas.values())})
        st.download_button(t("beta_download"), beta_df.to_csv(index=False).encode("utf-8-sig"), "beta.csv", "text/csv")

# フッター
st.markdown(f"""
<hr style="margin-top: 3rem; margin-bottom: 1rem; border: none; border-top: 1px solid #444;">
<div style='text-align: left; font-size: 0.8rem; color: gray;'>{t("disclaimer")}</div>
""", unsafe_allow_html=True)
