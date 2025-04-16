import streamlit as st
import pandas as pd
import yfinance as yf
import unicodedata
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io

# ページ設定とスタイル
st.set_page_config(page_title="最小分散フロンティアの計算", layout="wide")
st.markdown("""
    <style>
    .block-container {
        background-color: #000000;
        color: #ffffff;
        font-family: Meiryo, sans-serif;
    }
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
    header, footer, .stActionButton, .stDeployButton, .st-emotion-cache-13ln4jf, .st-emotion-cache-1avcm0n {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# セッション初期化
if 'calculating' not in st.session_state:
    st.session_state.calculating = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'added_buttons' not in st.session_state:
    st.session_state.added_buttons = set()

# 銘柄リスト取得
@st.cache_data
def load_japan_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url, skiprows=1, header=None)
    df = df[[1, 2]].dropna()
    df.columns = ['コード', '銘柄名']
    df['コード'] = df['コード'].astype(str).str.strip()
    return df

stock_df = load_japan_stock_list()

# 補助関数
def normalize_input(text):
    return unicodedata.normalize('NFKC', text)

def get_japanese_name(code):
    match = stock_df[stock_df['コード'].astype(str).str.upper() == str(code).upper()]
    if not match.empty:
        return match['銘柄名'].values[0]
    return None

# タイトル
st.markdown("<h1 style='text-align: center;'>最小分散フロンティアの計算</h1>", unsafe_allow_html=True)

# 入力方式選択
input_mode = st.radio("データ入力方法を選択してください。", ["証券コード・銘柄名による入力", "CSVによる入力"], horizontal=False)
use_csv = input_mode == "CSVによる入力"

uploaded_file = None
log_returns = None
csv_mode = False

# ========== CSV入力モード ==========
if use_csv:
    st.subheader("CSVファイルをアップロード")
    uploaded_file = st.file_uploader("CSVファイルを選択", type="csv")

    sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=10, freq="B")
    template_data = pd.DataFrame({
        date.strftime("%Y-%m-%d"): [
            np.random.randint(2000, 2500),
            np.random.randint(10000, 12000),
            np.random.randint(3000, 3500)
        ] for date in sample_dates
    }, index=["7203", "6758", "9432"])
    template_data.index.name = ""
    csv_buffer = io.StringIO()
    template_data.to_csv(csv_buffer)
    csv_data = csv_buffer.getvalue()

    st.markdown("<div class='template-button-container'>", unsafe_allow_html=True)
    st.download_button("CSVテンプレートをダウンロード", data=csv_data, file_name="template.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file, index_col=0)
        try:
            df_csv.columns = pd.to_datetime(df_csv.columns)
        except Exception as e:
            st.error(f"日付のパースに失敗しました: {e}")
        df_csv_display = df_csv.copy()
        df_csv_display.columns = df_csv_display.columns.strftime('%Y/%m/%d')
        st.write("CSVのプレビュー")
        st.dataframe(df_csv_display)
        csv_mode = True
        log_returns = np.log(df_csv / df_csv.shift(axis=1, periods=1)).dropna(axis=1)

    min_weight = st.number_input("最小投資割合", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    num_steps = st.number_input("期待利益率の段階数", min_value=5, max_value=100, value=50, step=1)

# ========== 証券コード・銘柄名入力モード ==========
if not use_csv:
    st.subheader("銘柄を検索して追加")
    search_type = st.radio("検索方法を選んでください。", ["証券コードで検索", "銘柄名で検索"])
    input_value = st.text_input("検索キーワードを入力してください。")
    input_value = normalize_input(input_value)

    if input_value:
        if search_type == "証券コードで検索":
            code = input_value.upper().replace(".T", "")
            name = get_japanese_name(code)
            if not name:
                st.warning("証券コードが見つかりません")
            else:
                if code not in [s['code'] for s in st.session_state.selected_stocks]:
                    if st.button(f"{code} ({name}) を追加", key=f"add_{code}"):
                        st.session_state.selected_stocks.append({"code": code, "name": name})
        else:
            matches = stock_df[stock_df['銘柄名'].str.contains(input_value, case=False, na=False)]
            if matches.empty:
                st.warning("銘柄名が見つかりません。")
            else:
                selected_name = st.radio("候補から選択", options=matches['銘柄名'].tolist())
                selected_code = matches[matches['銘柄名'] == selected_name]['コード'].values[0]
                if selected_code not in [s['code'] for s in st.session_state.selected_stocks]:
                    if st.button(f"{selected_code} ({selected_name}) を追加", key=f"add_{selected_code}"):
                        st.session_state.selected_stocks.append({"code": selected_code, "name": selected_name})

    if st.session_state.selected_stocks:
        st.subheader("選択中の銘柄リスト")
        for i, stock in enumerate(st.session_state.selected_stocks):
            col1, col2, col3 = st.columns([2, 4, 1])
            col1.write(stock["code"])
            col2.write(stock["name"])
            with col3:
                if st.button("削除", key=f"del_{i}"):
                    st.session_state.selected_stocks.pop(i)
                    st.rerun()
        if st.button("リセット", key="reset", help="すべての選択をクリア", type="secondary"):
            st.session_state.selected_stocks = []
            st.session_state.result_data = None  # ← この行を追加
            st.rerun()

        st.markdown("### 計算条件の設定")
        def_date_end = date.today() - timedelta(days=1)
        def_date_start = def_date_end - timedelta(days=365)
        start_date = st.date_input("開始日", value=def_date_start)
        end_date = st.date_input("終了日", value=def_date_end)
        span = st.radio("スパン（日間・週間・月間）", ["日間", "週間", "月間"])
        interval_map = {"日間": "1d", "週間": "1wk", "月間": "1mo"}
        interval = interval_map[span]

        min_weight = st.number_input("最小投資割合", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        num_steps = st.number_input("期待利益率の段階数", min_value=5, max_value=100, value=50, step=1)

# ========== 計算ボタンと実行 ==========
if (use_csv and log_returns is not None) or (not use_csv and len(st.session_state.selected_stocks) > 1):
    if st.button("計算を実行", key="calc_button"):
        st.session_state.calculating = True

        if use_csv:
            df = log_returns.copy()
            tickers = df.index.tolist()
            mean_returns = df.mean(axis=1).values
            std_devs = df.std(axis=1, ddof=0).values
            cov_matrix = np.cov(df.values)
        else:
            tickers = [s['code'] for s in st.session_state.selected_stocks]
            log_returns = []
            for ticker in tickers:
                df = yf.download(ticker + ".T", start=start_date, end=end_date, interval=interval)
                if "Close" in df.columns and len(df) > 1:
                    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
                    log_returns.append(df["LogReturn"].dropna().values)
            log_returns = np.array([r for r in log_returns if len(r) == len(log_returns[0])])
            tickers = tickers[:len(log_returns)]
            mean_returns = np.mean(log_returns, axis=1)
            std_devs = np.std(log_returns, axis=1, ddof=0)
            cov_matrix = np.cov(log_returns)

        cov_matrix += np.eye(len(cov_matrix)) * 1e-10
        N = len(mean_returns)
        sum_r = np.sum(mean_returns)
        max_r = np.max(mean_returns)
        min_r = np.min(mean_returns)
        max_weight = 1 - min_weight * (N - 1)
        max_return = max_r * max_weight + (sum_r - max_r) * min_weight
        min_return = min_r * max_weight + (sum_r - min_r) * min_weight
        epsilon = 1e-6
        target_returns = np.linspace(min_return + epsilon, max_return - epsilon, int(num_steps))

        def calculate_portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        frontier_vol = []
        frontier_weights = []
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
            )
            bounds = tuple((min_weight, 1.0) for _ in range(N))
            init_guess = np.array([1/N] * N)
            result = minimize(
                calculate_portfolio_volatility,
                init_guess,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if result.success:
                frontier_vol.append(result.fun)
                frontier_weights.append(result.x)

        st.session_state.result_data = {
            "tickers": tickers,
            "mean_returns": mean_returns,
            "std_devs": std_devs,
            "target_returns": target_returns,
            "frontier_vol": frontier_vol,
            "frontier_weights": frontier_weights
        }
        st.session_state.calculating = False

# ========== 結果表示 ==========
if st.session_state.calculating:
    st.markdown("<span style='color: red; font-weight: bold;'>計算中...</span>", unsafe_allow_html=True)

if st.session_state.result_data:
    data = st.session_state.result_data
    st.subheader("各銘柄の標準偏差と期待利益率")
    df_mean = pd.DataFrame({
        "証券コード": data["tickers"],
        "標準偏差": data["std_devs"],
        "期待利益率": data["mean_returns"]
    })
    st.dataframe(df_mean.style.format({"標準偏差": "{:.5f}", "期待利益率": "{:.5f}"}), hide_index=True)

    st.subheader("最小分散フロンティア")
    fig, ax = plt.subplots()
    ax.plot(data["frontier_vol"], data["target_returns"], linestyle='-')
    min_index = np.nanargmin(data["frontier_vol"])
    x = data["frontier_vol"][min_index]
    y = data["target_returns"][min_index]
    ax.scatter(x, y, color='red', label='Minimum Risk Portfolio')
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    st.subheader("各期待利益率における標準偏差と投資割合")
    weight_df = pd.DataFrame(data["frontier_weights"], columns=data["tickers"])
    weight_df.insert(0, "標準偏差", data["frontier_vol"])
    weight_df.insert(1, "期待利益率", data["target_returns"])
    weight_df = weight_df.sort_values(by="期待利益率", ascending=False)
    st.dataframe(weight_df.style.format("{:.5f}"), use_container_width=True, hide_index=True)

    csv = weight_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSVとしてダウンロード",
        data=csv,
        file_name="frontier_weights.csv",
        mime="text/csv"
    )
