# ====必要なライブラリのインポート====

# streamlit(WebアプリのUI作成)
import streamlit as st

# データ操作ライブラリ(表形式データの処理)
import pandas as pd
import numpy as np

# 株価データの取得(Yahoo Finance API)
import yfinance as yf

# 日本語テキストの正規化
import unicodedata

# Webデータ取得関連(TONA取得などで使用)
import requests
import re
from bs4 import BeautifulSoup

# 数学・統計処理(最小分散フロンティアの計算)
from scipy.optimize import minimize

# 日付操作(前営業日の取得など)
from datetime import datetime, date, timedelta

# プロット用ライブラリ
import matplotlib.pyplot as plt

# CSVテンプレート生成などに使用
import io

# 日本の祝日判定ライブラリ(jpholiday)
import jpholiday


# ====初期化====

close_data = None  # 株価終値のDaraFrame
rf_rate_span = None  # 無リスク利子率(スパン単位換算値)

# ====ページ設定とカスタムスタイル====

# Webアプリのタイトルとレイアウト方向を設定
st.set_page_config(
    page_title="最小分散フロンティアの計算",  # ブラウザのタブに表示されるタイトル
    layout="centered"  # 表示の中央寄せ
)

# アプリの背景やボタン等の見た目をCSSで変更
st.markdown("""
    <style>
    .block-container {
        background-color: #000000;  /* 背景を黒に */
        color: #ffffff;  /* 文字色を白に */
        font-family: Meiryo, sans-serif;  /* メイリオフォントを使用 */
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

# Streamlitアプリでは、セッションごとに変数を保持できる「st.session_state」を使って状態管理を行う。
# 各変数が未定義の場合に初期値を設定する。

if 'calculating' not in st.session_state:
    st.session_state.calculating = False  # 計算処理中であることを表すフラグ

if 'result_data' not in st.session_state:
    st.session_state.result_data = None  # 計算結果を格納する変数

if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []  # ユーザが選択した銘柄のリスト

if 'added_buttons' not in st.session_state:
    st.session_state.added_buttons = set()  # 動的に作成されたボタンのIDを保存(重複防止などに使用)


# ====日本株の銘柄リストを取得する関数====

@st.cache_data  # Streamlitのキャッシュデコレーター(毎回ダウンロードせずに済む)
def load_japan_stock_list():
    """
    日本取引所(JPX)が公開している銘柄リスト(Excel)を読み込み、
    証券コードと銘柄名の一覧をDataFrameとして返す。
    読み込みに失敗した場合はNoneを返す。
    """
    # 東京証券取引所が提供する東証上場銘柄一覧(Excel)
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    
    try:
        # Excelを読み込む(1行目をスキップ)
        df = pd.read_excel(url, skiprows=1, header=None)

        #必要な列(1列目:コード、2列目:銘柄名)のみ抽出し、欠損行を除外
        df = df[[1, 2]].dropna()

        # 列名を設定(コードと銘柄名)
        df.columns = ['コード', '銘柄名']

        # コードを文字列に変換して空白除去
        df['コード'] = df['コード'].astype(str).str.strip()

        return df
    
    except Exception as e:
        # エラー発生時は画面に警告表示し、Noneを返す
        st.error(f"銘柄リストの取得に失敗しました。ネット接続またはJPXのページをご確認ください。 {e}")
        return None

# 上で定義した関数を実行し、結果を変数に保持
stock_df = load_japan_stock_list()
if stock_df is None:
    st.stop()  # 銘柄リストが取得できなかった場合は処理を中断


# ====ユーザー入力の正規化関数====

def normalize_input(text):
    """
    入力文字列を正規化(全角→半角、記号の統一など)して返す。
    例: '７２０３' → '7203'
    """
    return unicodedata.normalize('NFKC', text)


# ====証券コードから銘柄名を取得====

def get_japanese_name(code):
    """
    証券コードから対応する銘柄名を検索して返す。
    一致しない場合はNoneを返す。
    """
    try:
        match = stock_df[stock_df['コード'].astype(str).str.upper() == str(code).upper()]
        if not match.empty:
            return match['銘柄名'].values[0]
        return None
    except Exception as e:
        st.error(f"銘柄名の取得中にエラーが発生しました：{e}")
        return None


# ====タイトル表示(カスタムCSSで装飾)====

# 通常表示では大きく、モバイル表示では小さく表示されるように調整
st.markdown("""
    <style>
    /* タイトルの基本スタイル */
    .custom-title {
        text-align: center;
        font-size: 40px;
        margin-bottom: 1rem;
    }

    /* モバイル表示（画面幅600px以下）ではフォントサイズを縮小 */
    @media screen and (max-width: 600px) {
        .custom-title {
            font-size: 25px;
        }
    }
    </style>
    
    <!-- タイトルを表示する要素 -->
    <div class='custom-title'>最小分散フロンティアの計算</div>
""", unsafe_allow_html=True)

# 区切り線を表示
st.markdown("""
<hr style='border: 1px solid white; margin: 25px 0;' />
""", unsafe_allow_html=True)


# ====入力方式選択====

# ユーザにデータ入力の方式(CSV or 証券コード・銘柄名)を選ばせる
# 前回の入力方式をセッションから取得(初回はNone)
previous_input_mode = st.session_state.get("previous_input_mode", None)

# ユーザに入力方式を選ばせるラジオボタン
input_mode = st.radio(
    "データ入力方法を選択してください。", 
    ["証券コード・銘柄名による入力", "CSVによる入力"], 
    horizontal=False
)

# 入力方式が前回と異なる場合、セッション情報を初期化(結果などをクリア)
if "previous_input_mode" not in st.session_state:
    st.session_state.previous_input_mode = input_mode

if st.session_state.previous_input_mode != input_mode:
    st.session_state.close_df = None
    st.session_state.df_csv = None
    st.session_state.log_returns = None
    st.session_state.result_data = None
    st.session_state.calculating = False
    st.session_state.previous_input_mode = input_mode

# 現在の入力方式をセッションに記録(次回の比較に使用)
st.session_state.previous_input_mode = input_mode

# CSV入力モードを真偽値で保持(以降の分岐処理に使用)
use_csv = input_mode == "CSVによる入力"

# 区切り線を表示
st.markdown("---")


# ====入力データ関連の初期化====

uploaded_file = None  # ユーザがアップロードするCSVファイルを格納
log_returns = None  # 株価データから計算されるログリターン(DataFrameまたは配列)


# ====TONA金利（無担保コール翌日物）を取得する関数====

def get_previous_business_day(offset=1):
    """
    現在から指定した日数だけ前の「日本の営業日」(土日・祝日を除く)を返す。
    例: offset=1 → 直近の営業日
    """
    day = datetime.now().date() - timedelta(days=offset)
    while day.weekday() >= 5 or jpholiday.is_holiday(day):  # 土日または祝日
        offset += 1
        day = datetime.now().date() - timedelta(days=offset)
    return day

def fetch_tona_rate_report():
    """
    日本銀行のコール市場関連統計ページから、TONA(無担保コールＯ／Ｎ物レート)の平均値を取得。
    成功すれば、(金利のfloat、対象日付、URL)を返す。失敗時は(None、None、URL)を返す。
    """
    now = datetime.now()
    # 10時より前なら2営業日前、以降なら1営業日前を使う(通常午前10時頃公開)
    target_date = get_previous_business_day(offset=2 if now.hour < 10 else 1)
    url_date = target_date.strftime("%y%m%d")
    url = f"https://www3.boj.or.jp/market/jp/stat/md{url_date}.htm"

    try:
        response = requests.get(url, timeout=10)
        response.encoding = "shift_jis"  # 日本銀行ページはshift_jis
        soup = BeautifulSoup(response.text, "html.parser")

        # 金利と日付を<li>タグの中から取得
        li_tags = soup.find_all("li")
        for li in li_tags:
            if "平均" in li.text:
                # 金利を数値で取得
                match_rate = re.search(r"(\d+\.\d+)", li.text)

                # 日付も抽出
                span = li.find("span", class_=re.compile(r"projection call"))
                match_date = re.search(r"\d{4}/\d{1,2}/\d{1,2}", " ".join(span.get("class", []))) if span else None

                if match_rate and match_date:
                    rate = float(match_rate.group(1))
                    tona_date = match_date.group(0)
                    return rate, tona_date, url

        print("TONAのレートが見つかりませんでした。")
        return None, None, url

    except Exception as e:
        print(f"TONA取得エラー: {e}")
        return None, None, url


# ====株価終値を一括取得する関数====

def fetch_close_prices(codes, start_date, end_date, interval):
    """
    与えられた証券コードリストに対して、指定期間・頻度の株価終値を一括取得する。
    """
    import yfinance as yf
    tickers = [code + ".T" for code in codes]
    df = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval=interval,
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )
    
    price_data = {}
    for code in codes:
        ticker = code + ".T"
        try:
            close = df[ticker]["Close"]
            price_data[code] = close
        except (KeyError, TypeError):
            price_data[code] = None
    return price_data


# ====株価終値からログリターンを計算する関数====

def calculate_log_returns(df, axis="auto"):
    """
    指定された方向に沿ってログリターンを計算する。
    - axis=0 : 列方向（通常、行＝日付、列＝銘柄）
    - axis=1 : 行方向（通常、行＝銘柄、列＝日付）
    - axis='auto' の場合は列名が日付型なら axis=1、それ以外は axis=0 に自動判定
    """

    if df.isnull().values.any():
        raise ValueError("欠損値が含まれています。")
    if (df <= 0).values.any():
        raise ValueError("0以下の株価が含まれています。")
    if df.shape[1] < 2:
        raise ValueError("日付列が2列以上必要です。")

    df = df.sort_index()  # 日付順にソート

    # axis 自動判定：datetime型がどちらにあるかで決定
    if axis == "auto":
        if pd.api.types.is_datetime64_any_dtype(df.columns):
            axis = 1
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            axis = 0
        else:
            raise ValueError("時系列（日付）情報が index/columns のどちらにも見つかりません。")

    return np.log(df / df.shift(axis=axis)).dropna(axis=axis)


# ====CSV入力モード====

if use_csv:
    # span及びintervalを初期化
    span = None
    interval = None

    # ボタン位置の整理
    col_csv, col_button = st.columns([5, 2])
    with col_csv:
        # CSVファイルのアップロード
        uploaded_file = st.file_uploader("CSVファイルを選択", type="csv")

        # サンプルCSVの作成
        # 本日を含めた直近10営業日(平日(祝日は考慮しない))の日付のリストを作成
        sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=10, freq="B")
        # ダミーの株価データを作成
        template_data = pd.DataFrame({
            date.strftime("%Y-%m-%d"): [  # 各日付を"YYYY-MM-DD"形式にして列に
                np.random.randint(2500, 2750),  # 7203用(例:トヨタ自動車)
                np.random.randint(3250, 3750),  # 6758用(例:ソニーグループ)
                np.random.randint(140, 150)  # 9432用(例:日本電信電話)
            ] for date in sample_dates  # 各営業日毎に値を生成
        }, index=["7203", "6758", "9432"])  # 証券コードを行に
        # 1行目のインデックス名を空にする
        template_data.index.name = ""
        # CSV化(文字列としてメモリ上に保存)
        csv_buffer = io.StringIO()
        template_data.to_csv(csv_buffer)
        # CSV文字列を取り出す
        csv_data = csv_buffer.getvalue()

    with col_button:
        # サンプルCSVのダウンロード
        st.download_button("サンプルCSVをDL", data=csv_data, file_name="sample.csv", mime="text/csv")
        
    # 実際のCSVファイル処理
    if uploaded_file:
        try:
            # CSVの読み込み(index_col=0で証券コードをインデックスに)
            df_csv = pd.read_csv(uploaded_file, index_col=0)

            # 日付列の変換(文字列→datetime)
            parsed_dates = pd.to_datetime(df_csv.columns, errors='coerce')
            if parsed_dates.isnull().any():
                raise ValueError("日付列の形式が不正です。")
            df_csv.columns = parsed_dates  # 日付型に変換

            # 必ず日付順にソート（念のため）
            df_csv = df_csv.sort_index()

            # スパン(日/週/月)の推定(平均間隔による)
            dates = df_csv.columns
            try:
                deltas = np.diff(dates).astype('timedelta64[D]').astype(int)
                avg_delta = np.mean(deltas)
                if avg_delta <= 2:
                    span = "日間"
                elif avg_delta <= 10:
                    span = "週間"
                else:
                    span = "月間"
                st.info(f"スパンは自動判定されました：**{span}**（平均間隔 {avg_delta:.1f} 日）")
            except Exception as e:
                st.error(f"スパンの推定中にエラーが発生しました：{e}")
                st.stop()

            # CSVプレビュー表示
            df_csv_display = df_csv.copy()
            try:
                df_csv_display.columns = [col.strftime('%Y/%m/%d') for col in df_csv_display.columns]
            except Exception as e:
                st.error(f"列名の日付表示の整形に失敗しました（{e}）。")
            with st.expander("CSVのプレビューを表示"):
                st.dataframe(df_csv_display)

            # 株価データをセッションに保存
            st.session_state.df_csv = df_csv  # 読み込んだ株価（行：銘柄、列：日付）

        except Exception as e:
            st.error(f"CSVのデータ処理中にエラーが発生しました。:{e}")


# ====証券コード・銘柄名入力モード====

if not use_csv:
    # span及びrf_rate_spanを初期化
    span = None
    rf_rate_span = None

    search_type = st.radio("検索方法を選んでください。", ["証券コードで検索", "銘柄名で検索"])
    input_value = normalize_input(st.text_input("検索キーワードを入力してください。"))

    if input_value:
        if search_type == "証券コードで検索":
            matches = stock_df[stock_df['コード'].str.contains(input_value, case=False, na=False)]
        else:
            matches = stock_df[stock_df['銘柄名'].str.contains(input_value, case=False, na=False)]

        if matches.empty:
            st.warning("該当する銘柄が見つかりませんでした。")
        else:
            match_label = matches['コード'] + " " + matches['銘柄名']
            selected_label = st.radio("追加したい銘柄を選択", options=match_label.tolist(), key="radio_candidates")

            # コードと銘柄名に分割
            selected_code, selected_name = selected_label.split(" ", 1)

            # まだ追加されていない場合のみ追加ボタンを表示
            if selected_code not in [s['code'] for s in st.session_state.selected_stocks]:
                if st.button(f"{selected_code}（{selected_name}）を追加"):
                    st.session_state.selected_stocks.append({"code": selected_code, "name": selected_name})
            else:
                st.info(f"{selected_code}（{selected_name}）はすでに選択されています。")

    # 選択済みの銘柄の表示・削除・リセット
    if st.session_state.selected_stocks:
        st.markdown("<p style='font-size: 20px; font-weight: normal;'>選択中の銘柄リスト</p>", unsafe_allow_html=True)
        for i, stock in enumerate(st.session_state.selected_stocks):
            cols = st.columns([2, 4, 1])
            cols[0].write(stock["code"])  # 証券コードを表示
            cols[1].write(stock["name"])  # 銘柄名を表示
            if cols[2].button("削除", key=f"del_{i}"):  # 削除ボタンを表示
                st.session_state.selected_stocks.pop(i)
                st.rerun()

        if st.button("リセット", key="reset", help="すべての選択をクリア", type="secondary"):
            st.session_state.selected_stocks = []
            st.session_state.result_data = None  
            st.rerun()

        # 区切り線を表示
        st.markdown("---")

        # 日付とスパンの設定
        def_date_end = date.today() - timedelta(days=1)
        def_date_start = def_date_end - timedelta(days=365)
        start_date = st.date_input("開始日", value=def_date_start)
        end_date = st.date_input("終了日", value=def_date_end)

        if end_date > date.today():
            st.error("終了日が未来の日付になっています。正しい終了日を選んでください。")
            st.stop()
        elif start_date >= end_date:
            st.error("開始日は終了日より前の日付を選択してください。")
            st.stop()

        span = st.radio("スパン（日間・週間・月間）", ["日間", "週間", "月間"])
        interval_map = {"日間": "1d", "週間": "1wk", "月間": "1mo"}
        interval = interval_map[span]

        # 株価一括取得
        codes = [s["code"] for s in st.session_state.selected_stocks]
        price_data = fetch_close_prices(codes, start_date, end_date, interval)

        # 共通日付でDataFrame整形
        close_df = pd.DataFrame({
            code: data for code, data in price_data.items() if data is not None
        }).dropna(axis=1)

        if close_df.empty:
            st.error("株価データの取得に失敗しました。")
            st.stop()

        # ソート（念のため）
        close_df = close_df.sort_index()

        # 株価DataFrameをセッションに保存
        st.session_state.close_df = close_df


# ====最小投資割合と期待利益率の段階数の入力====

min_weight = st.number_input("最小投資割合", min_value=0.0, max_value=0.5, value=0.01, step=0.001, format="%.3f")
num_steps = st.number_input("期待利益率の段階数", min_value=5, max_value=500, value=50, step=1)


# ====TONA取得と初期値の決定====

rate, tona_date, tona_url = fetch_tona_rate_report()

if rate and tona_date:  # TONAの取得に成功した場合
    st.info(f"{tona_date} のTONAは {rate:.3f}% です。\n[日本銀行]({tona_url})")
    rf_rate_default = rate
else:  # TONAの取得に失敗した場合
    st.warning(f"TONAの取得に失敗しました。\n[日本銀行]({tona_url}) を確認してください。")
    rf_rate_default = 0.48  # デフォルト値

# ユーザが任意に設定可能な入力フォーム(%で表示)
rf_rate = st.number_input(
    "無リスク資産の年利（%）", 
    min_value=0.0, max_value=100.0,  
    value=rf_rate_default, 
    step=0.001, 
    format="%.3f"
) / 100  # 実際の計算では少数を使う

# rf_rate(TONA(年利))をスパン単位に変換（定義のみ）
def convert_rf_rate_safe(rf_rate, span):
    """
    年率のリスクフリーレートをスパン（日間・週間・月間）に応じて変換。
    spanが不明な場合はエラーとして処理を停止。
    """
    if span == "日間":
        return rf_rate / 252
    elif span == "週間":
        return rf_rate / 52
    elif span == "月間":
        return rf_rate / 12
    else:
        st.error(f"スパンが未定義または不正です（スパン：{span}）。")
        st.stop()

# span(日/週/月)が決まったら、rf_rate(年利)をスパン単位に変換
if span:
    rf_rate_span = convert_rf_rate_safe(rf_rate, span)


# ====計算ボタンと実行====

# 計算ボタンを表示する条件
# CSVモードの場合: CSVファイルがアップロードされており、df_csvが存在する
# 証券コードモードの場合: close_dfが存在し、2銘柄以上が選択されている
show_calc_button = (
    (use_csv and "df_csv" in st.session_state and st.session_state.df_csv is not None and uploaded_file is not None) or
    (not use_csv and "close_df" in st.session_state and st.session_state.close_df is not None and len(st.session_state.selected_stocks) >= 2)
)
# 計算ボタンの描画(押下時には run_calc がTrueになる)
if show_calc_button:
    run_calc = st.button("計算を実行", key="calc_button", disabled=st.session_state.calculating)
    
    # 計算開始時の準備(calculatingフラグを立て、画面上に「計算中です。」と表示)
    if run_calc:
        st.session_state.calculating = True
        
        with st.spinner("計算中です。"):
            if use_csv:  # CSV入力モード
                df = st.session_state.get("df_csv", None)
                if df is None:
                    st.error("CSVファイルのデータが見つかりません。")
                    st.stop()
                try:
                    log_returns = calculate_log_returns(df, axis=1)  # df_csvからログリターン計算(axis=1で行方向に時系列計算)
                except Exception as e:
                    st.error(f"ログリターンの計算に失敗しました：{e}")
                    st.stop()
                # 証券コードリスト
                tickers = log_returns.index.tolist()
                log_returns = log_returns.T  # 計算後、log_returnsを転置して「行＝日付、列＝銘柄」に揃える
                
            else:  # 証券コード・銘柄名入力モード
                df = st.session_state.get("close_df", None)
                if df is None:
                    st.error("株価データが見つかりません。")
                    st.stop()
                try:
                    log_returns = calculate_log_returns(df, axis=0)  # close_dfからログリターン計算(axis=0で列方向に時系列計算)
                except Exception as e:
                    st.error(f"ログリターンの計算に失敗しました：{e}")
                    st.stop()

                # 有効な銘柄数チェック（2銘柄未満なら中断）
                if log_returns.shape[1] < 2:
                    st.error("有効なデータを持つ銘柄が2つ以上必要です。期間を見直してください。")
                    st.session_state.calculating = False
                    st.stop()

                # 日数のカウント（有効な共通データ行数）
                valid_days = log_returns.shape[0]

                # 想定されるスパン単位の日数
                expected_days = (end_date - start_date).days
                expected_count = {
                    "日間": expected_days,
                    "週間": expected_days // 7,
                    "月間": expected_days // 30
                }[span]

                # 警告表示
                if valid_days < expected_count * 1:
                    st.info(
                        f"指定された期間（{expected_count}{span}）に対し、"
                        f"共通の有効株価データが存在するのは {valid_days}{span} のみです。"
                    )
                # 証券コードリスト
                tickers = log_returns.columns.tolist()

            # log_returnsをセッションに保存
            st.session_state.log_returns = log_returns
            # 平均・標準偏差・共分散を計算（列＝銘柄方向）
            mean_returns = log_returns.mean(axis=0).values
            std_devs = log_returns.std(axis=0, ddof=0).values
            cov_matrix = np.cov(log_returns.T.values)

            # 最小投資割合のバリデーションチェック(全銘柄均等以上の最小割合は許容しない)
            N = len(tickers)
            if min_weight >= (1 / N):
                st.error(f"選択された銘柄数に対して最小投資割合が大きすぎます（{1/N:.4f} 未満である必要があります）。")
                st.session_state.calculating = False
                st.stop()

            if min_weight < 0 or min_weight >= 0.5:
                st.error("最小投資割合は0以上0.5未満である必要があります。")
                st.session_state.calculating = False
                st.stop()

            # 株価時系列の表示
            if close_data is not None and not close_data.empty:
                with st.expander("株価の時系列（終値）を表示"):
                    close_data_display = close_data.T  # 銘柄を行、日付を列にする
                    close_data_display.columns = close_data_display.columns.strftime('%Y/%m/%d')  # 日付フォーマット変更
                    st.dataframe(close_data_display.round(2), use_container_width=True)

            cov_matrix += np.eye(len(cov_matrix)) * 1e-10
            N = len(mean_returns)
            sum_r = np.sum(mean_returns)
            max_r = np.max(mean_returns)
            min_r = np.min(mean_returns)
            max_weight = 1 - min_weight * (N - 1)
            max_return = max_r * max_weight + (sum_r - max_r) * min_weight
            min_return = min_r * max_weight + (sum_r - min_r) * min_weight
            epsilon = 1e-6
            # 指定された段階数に応じてリターン目標値を等間隔に設定
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

            if len(frontier_vol) == 0:
                st.error("最小分散フロンティアの計算に失敗しました。銘柄数・期間・最小投資割合を見直してください。")
                st.session_state.calculating = False
                st.stop()

            st.session_state.result_data = {
                "tickers": tickers,
                "mean_returns": mean_returns,
                "std_devs": std_devs,
                "cov_matrix": cov_matrix,
                "target_returns": target_returns,
                "frontier_vol": frontier_vol,
                "frontier_weights": frontier_weights
            }
        st.session_state.calculating = False


# ====結果表示====

if st.session_state.result_data:
    data = st.session_state.result_data
    # 銘柄ごとのリスク(log_returns標準偏差)・リターン(log_returns平均)情報を表示
    with st.expander("各銘柄の標準偏差と期待利益率を表示"):
        df_mean = pd.DataFrame({
            "証券コード": data["tickers"],
            "標準偏差": data["std_devs"],
            "期待利益率": data["mean_returns"]
        })
        st.dataframe(df_mean.style.format({"標準偏差": "{:.5f}", "期待利益率": "{:.5f}"}), hide_index=True)

    # 相関行列の可視化（ヒートマップ）
    with st.expander("銘柄間の相関行列（ヒートマップ）を表示"):
        # ログリターンをセッションから取得
        log_returns = st.session_state.get("log_returns", None)
        # 取得できない場合は警告を出して終了
        if data and log_returns is not None:
            try:
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
                corr_csv = corr_matrix.round(5).to_csv(index=True, encoding="utf-8-sig")
                st.download_button(
                    label="相関係数をCSVとしてダウンロード",
                    data=corr_csv,
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"相関行列の表示中にエラーが発生しました。: {e}")
        else:
            st.warning("相関行列を表示するには、まず計算を実行してください。")
            st.stop()

    with st.expander("最小分散フロンティアを表示"):

        if data["frontier_vol"] is None or len(data["frontier_vol"]) == 0:
            st.error("最小分散フロンティアが正常に計算できませんでした。")
        else:
            # 最小分散点のインデックス
            min_index = np.nanargmin(data["frontier_vol"])

            # 効率的フロンティア
            efficient_vol = data["frontier_vol"][min_index:]
            efficient_returns = data["target_returns"][min_index:]

            # シャープレシオ最大点（CML接点）
            sharpe_ratios = (np.array(data["target_returns"]) - rf_rate_span) / np.array(data["frontier_vol"])
            max_sharpe_idx = np.nanargmax(sharpe_ratios)
            max_std = data["frontier_vol"][max_sharpe_idx]
            max_return = data["target_returns"][max_sharpe_idx]

            # 資本市場線（CML）
            cml_x = np.linspace(0, max_std * 1, 100)
            cml_y = rf_rate_span + ((max_return - rf_rate_span) / max_std) * cml_x

            # グラフ描画
            fig, ax = plt.subplots(facecolor='black')
            ax.set_facecolor('black')

            # 最小分散フロンティア
            ax.plot(data["frontier_vol"], data["target_returns"], linestyle='-', color='gray', label='Minimum Variance Frontier', zorder=1)

            # 効率的フロンティア
            ax.plot(efficient_vol, efficient_returns, linestyle='-', color='cyan', linewidth=2, label='Efficient Frontier', zorder=2)

            # 最小リスクポートフォリオ
            x = data["frontier_vol"][min_index]
            y = data["target_returns"][min_index]
            ax.scatter(x, y, color='red', label='Minimum Risk Portfolio', zorder=5)

            # CML描画
            ax.plot(cml_x, cml_y, linestyle='--', color='gold', linewidth=1.8, label='Capital Market Line', zorder=0)


            # Y軸に無リスク利子率を含むように調整
            ymin = min(rf_rate_span * 0.9, min(data["target_returns"]) * 0.9)
            ymax = max(data["target_returns"]) * 1.1
            ax.set_ylim(ymin, ymax)

            # 無リスク利子率の線を描画
            ax.axhline(rf_rate_span, linestyle=':', color='orange', linewidth=1, label=f"Risk-Free Rate ({rf_rate_span:.4f})")

            # ラベル・軸・凡例の見た目調整
            ax.set_xlabel("Standard Deviation", color='white')
            ax.set_ylabel("Expected Return", color='white')
            ax.tick_params(colors='white')
            legend = ax.legend(loc='lower right', facecolor='black', labelcolor='white')
            legend.get_frame().set_visible(False)

            st.pyplot(fig)

    with st.expander("各期待利益率における標準偏差と投資割合を表示"):
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

st.markdown("""
    <hr style="margin-top: 3rem; margin-bottom: 1rem; border: none; border-top: 1px solid #444;">
    <div style='text-align: left; font-size: 0.8rem; color: gray;'>
        本アプリは学習目的で作成されたものであり、投資判断への利用を想定したものではありません。<br> 利用によって生じたいかなる損害についても開発者は責任を負いかねます。
    </div>
""", unsafe_allow_html=True)

