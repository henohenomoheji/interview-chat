from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PAGE_TITLE = "分析グラフ"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cafe_pl_data.csv"


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """CSVを読み込み、年月を並び替えやすい形に整形"""
    df = pd.read_csv(path)
    df["年月"] = pd.to_datetime(dict(year=df["年"], month=df["月"], day=1), errors="coerce")
    df["年月表示"] = df["年月"].dt.strftime("%Y-%m")
    return df


def build_dual_axis_chart(
    df: pd.DataFrame,
    axis_col: str,
    left_cfg: dict,
    right_cfg: dict,
) -> go.Figure:
    """左右軸の設定を受け取り、右軸にタイトルを付けたグラフを生成する"""
    if axis_col == "年月表示":
        order = df.sort_values("年月")[axis_col].dropna().unique().tolist()
    elif pd.api.types.is_numeric_dtype(df[axis_col]):
        order = sorted(df[axis_col].dropna().unique().tolist())
    else:
        order = sorted(df[axis_col].astype(str).dropna().unique().tolist())

    # 色と線種を固定
    items = sorted(df["財務項目"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {item: palette[i % len(palette)] for i, item in enumerate(items)}

    versions = sorted(df["バージョン"].unique())
    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    dash_map = {v: dash_styles[i % len(dash_styles)] for i, v in enumerate(versions)}

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    order_map = {v: i for i, v in enumerate(order)}

    def add_traces(axis_df: pd.DataFrame, chart_type: str, secondary: bool, axis_name: str):
        if axis_df.empty:
            return

        grouped = axis_df.groupby([axis_col, "財務項目", "バージョン"], as_index=False)["値"].sum()
        grouped["順序"] = grouped[axis_col].map(order_map)
        grouped = grouped.sort_values("順序")

        for (item, ver), g in grouped.groupby(["財務項目", "バージョン"]):
            name = f"{item}｜{ver}（{axis_name}）"
            common = dict(
                x=g[axis_col],
                y=g["値"],
                name=name,
                legendgroup=f"{item}-{ver}-{axis_name}",
                marker_color=color_map[item],
            )

            if chart_type == "棒グラフ":
                trace = go.Bar(**common, opacity=0.7)
            elif chart_type == "折れ線グラフ":
                trace = go.Scatter(**common, mode="lines+markers", line=dict(dash=dash_map.get(ver, "solid")))
            else:  # 面グラフ
                trace = go.Scatter(
                    **common,
                    mode="lines",
                    line=dict(dash=dash_map.get(ver, "solid")),
                    fill="tozeroy",
                    fillcolor=color_map[item],
                    opacity=0.4,
                )

            fig.add_trace(trace, secondary_y=secondary)

    add_traces(right_cfg["data"], right_cfg["chart_type"], True, "軸右")
    add_traces(left_cfg["data"], left_cfg["chart_type"], False, "軸左")

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title="財務項目 / バージョン / 軸",
        barmode="group",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(title_text=left_cfg["label"], secondary_y=False)
    fig.update_yaxes(title_text=right_cfg["label"], secondary_y=True)
    return fig


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("Cafe PL データを使った財務指標の可視化")

# データ読み込み
df = load_data(DATA_PATH)

with st.sidebar:
    st.header("ページ選択")
    st.write("分析グラフ")

# 上：財務項目フィルタ
st.markdown("### 上：財務項目を選択")
financial_items = sorted(df["財務項目"].unique().tolist())
selected_financial = st.multiselect("財務項目（複数選択可）", options=financial_items, default=financial_items)

filtered = df.copy()
if selected_financial:
    filtered = filtered[filtered["財務項目"].isin(selected_financial)]

if filtered.empty:
    st.warning("選択条件に一致するデータがありません。条件を変更してください。")
    st.stop()

# 中：バージョン、軸ラベル、グラフ種類（軸右 / 軸左）
st.markdown("### 中：表示パラメータ")

version_options = sorted(filtered["バージョン"].unique().tolist())
y_axis_options = ["単価", "合計金額", "パーセンテージ"]
chart_type_options = ["棒グラフ", "面グラフ", "折れ線グラフ"]

st.markdown("#### 軸右")
r1, r2, r3 = st.columns(3)
right_versions = r1.multiselect("バージョン（軸右）", options=version_options, default=version_options, key="right_versions")
right_y_axis = r2.selectbox("軸ラベル（軸右）", y_axis_options, index=1, key="right_y_axis")
right_chart_type = r3.selectbox("グラフ種類（軸右）", chart_type_options, index=0, key="right_chart_type")

st.markdown("#### 軸左")
l1, l2, l3 = st.columns(3)
left_versions = l1.multiselect("バージョン（軸左）", options=version_options, default=version_options, key="left_versions")
left_y_axis = l2.selectbox("軸ラベル（軸左）", y_axis_options, index=0, key="left_y_axis")
left_chart_type = l3.selectbox("グラフ種類（軸左）", chart_type_options, index=0, key="left_chart_type")

left_df = (
    filtered[filtered["バージョン"].isin(left_versions) & (filtered["単位"] == left_y_axis)]
    if left_versions
    else pd.DataFrame(columns=filtered.columns)
)
right_df = (
    filtered[filtered["バージョン"].isin(right_versions) & (filtered["単位"] == right_y_axis)]
    if right_versions
    else pd.DataFrame(columns=filtered.columns)
)

if left_df.empty and right_df.empty:
    st.warning("選択条件に一致するデータがありません。条件を変更してください。")
    st.stop()

# 下：グラフ＋データ
st.markdown("### 下：可視化結果とデータ")
fig = build_dual_axis_chart(
    filtered,
    axis_col="年月表示",
    left_cfg={"data": left_df, "chart_type": left_chart_type, "label": left_y_axis},
    right_cfg={"data": right_df, "chart_type": right_chart_type, "label": right_y_axis},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("データ表示（先頭100件）")
st.dataframe(filtered.head(100), use_container_width=True, height=420)
