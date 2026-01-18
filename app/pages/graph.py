from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PAGE_TITLE = "分析グラフ"
#DATA_PATH = Path(__file__).resolve().parent.parent / "db" / "data" / "pl_data.csv"
DATA_PATH = "/workspaces/test-analysis/db/data/pl_data.csv"
HEATMAP_FIXED_VERSION = "当年"
# Avoid unreadable charts when points explode.
LABEL_THRESHOLD = 200


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """CSVを読み込み、年月を並び替えやすい形に整形"""
    df = pd.read_csv(path)
    df["年月"] = pd.to_datetime(dict(year=df["年"], month=df["月"], day=1), errors="coerce")
    df["年月表示"] = df["年月"].dt.strftime("%Y-%m")
    return df


def format_millions(value: float) -> str:
    return f"{value / 1_000_000:.1f}"


def add_value_labels(frame: pd.DataFrame, value_col: str, formatter=format_millions) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["ラベル"] = labeled[value_col].apply(formatter)
    return labeled


def should_show_labels(frame: pd.DataFrame) -> bool:
    return len(frame) <= LABEL_THRESHOLD


def legend_title_with_unit(title: str, unit: str = "M") -> str:
    return f"{title}（{unit}）"


def resolve_stack_by(
    stack_dim: str,
    selected_channel_lv1: str,
    selected_channel_lv2: str,
    selected_category_lv1: str,
    selected_category_lv2: str,
) -> str:
    if stack_dim == "財務項目":
        return "財務項目"
    if stack_dim == "チャネル":
        if selected_channel_lv1 == "-":
            return "channel_lv1"
        if selected_channel_lv2 == "-":
            return "channel_lv2"
        return "channel_lv3"
    if stack_dim == "カテゴリ":
        if selected_category_lv1 == "-":
            return "category_lv1"
        if selected_category_lv2 == "-":
            return "category_lv2"
        return "category_lv3"
    raise ValueError(stack_dim)


def build_month_version_order(
    frame: pd.DataFrame,
    version_order: list[str] | None = None,
    month_col: str = "年月表示",
    raw_month_col: str = "年月",
    version_col: str = "バージョン",
) -> list[str]:
    if frame.empty:
        return []
    month_order = frame.sort_values(raw_month_col)[month_col].dropna().unique().tolist()
    if version_order is None:
        version_order = sorted(frame[version_col].dropna().unique().tolist())
    existing = set(frame[month_col] + "｜" + frame[version_col])
    x_order = []
    for month in month_order:
        for version in version_order:
            key = f"{month}｜{version}"
            if key in existing:
                x_order.append(key)
    return x_order


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

    def add_traces(axis_df: pd.DataFrame, chart_type: str, secondary: bool, axis_name: str, labels_on: bool):
        if axis_df.empty:
            return

        grouped = axis_df.groupby([axis_col, "財務項目", "バージョン"], as_index=False)["値"].sum()
        grouped["順序"] = grouped[axis_col].map(order_map)
        grouped = grouped.sort_values("順序")
        show_labels = labels_on and should_show_labels(grouped)
        grouped = add_value_labels(grouped, "値")

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
                trace = go.Bar(
                    **common,
                    opacity=0.7,
                    text=g["ラベル"] if show_labels else None,
                    texttemplate="%{text}" if show_labels else None,
                    textposition="outside",
                )
            elif chart_type == "折れ線グラフ":
                trace = go.Scatter(
                    **common,
                    mode="lines+markers+text" if show_labels else "lines+markers",
                    line=dict(dash=dash_map.get(ver, "solid")),
                    text=g["ラベル"] if show_labels else None,
                    textposition="top center",
                )
            else:  # 面グラフ
                trace = go.Scatter(
                    **common,
                    mode="lines+text" if show_labels else "lines",
                    line=dict(dash=dash_map.get(ver, "solid")),
                    fill="tozeroy",
                    fillcolor=color_map[item],
                    opacity=0.4,
                    text=g["ラベル"] if show_labels else None,
                    textposition="top center",
                )

            fig.add_trace(trace, secondary_y=secondary)

    add_traces(right_cfg["data"], right_cfg["chart_type"], True, "軸右", right_cfg["show_labels"])
    add_traces(left_cfg["data"], left_cfg["chart_type"], False, "軸左", left_cfg["show_labels"])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title=legend_title_with_unit("財務項目 / バージョン / 軸"),
        barmode="group",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(title_text=left_cfg["label"], secondary_y=False)
    fig.update_yaxes(title_text=right_cfg["label"], secondary_y=True)
    return fig


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("PL データを使った財務指標の可視化")

# データ読み込み
df = load_data(DATA_PATH)

with st.sidebar:

    st.markdown("### フィルター")
    company_options = ["-"] + sorted(df["company"].dropna().unique().tolist())
    default_company = "Starbacks" if "Starbacks" in company_options else "-"
    selected_company = st.selectbox(
        "company",
        options=company_options,
        index=company_options.index(default_company),
        key="company",
    )
    min_date = df["年月"].min()
    max_date = df["年月"].max()
    period_range = st.slider(
        "期間",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM",
        step=pd.Timedelta(days=31),
    )

    channel_lv1_options = ["-"] + sorted(df["channel_lv1"].dropna().unique().tolist())
    selected_channel_lv1 = st.selectbox("channel_lv1", options=channel_lv1_options, index=0, key="channel_lv1")

    channel_lv2_base = df
    if selected_channel_lv1 != "-":
        channel_lv2_base = channel_lv2_base[channel_lv2_base["channel_lv1"] == selected_channel_lv1]
    channel_lv2_options = ["-"] + sorted(channel_lv2_base["channel_lv2"].dropna().unique().tolist())
    selected_channel_lv2 = st.selectbox("channel_lv2", options=channel_lv2_options, index=0, key="channel_lv2")

    channel_lv3_base = channel_lv2_base
    if selected_channel_lv2 != "-":
        channel_lv3_base = channel_lv3_base[channel_lv3_base["channel_lv2"] == selected_channel_lv2]
    channel_lv3_options = ["-"] + sorted(channel_lv3_base["channel_lv3"].dropna().unique().tolist())
    selected_channel_lv3 = st.selectbox("channel_lv3", options=channel_lv3_options, index=0, key="channel_lv3")

    category_lv1_options = ["-"] + sorted(df["category_lv1"].dropna().unique().tolist())
    selected_category_lv1 = st.selectbox(
        "category_lv1", options=category_lv1_options, index=0, key="category_lv1"
    )

    category_lv2_base = df
    if selected_category_lv1 != "-":
        category_lv2_base = category_lv2_base[category_lv2_base["category_lv1"] == selected_category_lv1]
    category_lv2_options = ["-"] + sorted(category_lv2_base["category_lv2"].dropna().unique().tolist())
    selected_category_lv2 = st.selectbox(
        "category_lv2", options=category_lv2_options, index=0, key="category_lv2"
    )

    category_lv3_base = category_lv2_base
    if selected_category_lv2 != "-":
        category_lv3_base = category_lv3_base[category_lv3_base["category_lv2"] == selected_category_lv2]
    category_lv3_options = ["-"] + sorted(category_lv3_base["category_lv3"].dropna().unique().tolist())
    selected_category_lv3 = st.selectbox(
        "category_lv3", options=category_lv3_options, index=0, key="category_lv3"
    )


filtered_df = df.copy()
start_date, end_date = period_range
filtered_df = filtered_df[
    (filtered_df["年月"] >= pd.to_datetime(start_date)) & (filtered_df["年月"] <= pd.to_datetime(end_date))
]

for col, selected in [
    ("company", selected_company),
    ("channel_lv1", selected_channel_lv1),
    ("channel_lv2", selected_channel_lv2),
    ("channel_lv3", selected_channel_lv3),
    ("category_lv1", selected_category_lv1),
    ("category_lv2", selected_category_lv2),
    ("category_lv3", selected_category_lv3),
]:
    if selected != "-":
        filtered_df = filtered_df[filtered_df[col] == selected]

if filtered_df.empty:
    st.warning("選択条件に一致するデータがありません。条件を変更してください。")
    st.stop()

y_axis_options = ["単価", "合計金額", "パーセンテージ"]
version_options = sorted(filtered_df["バージョン"].unique().tolist())
default_version = "当月" if "当月" in version_options else version_options[0]
financial_items = sorted(filtered_df["財務項目"].unique().tolist())

tab_line, tab_heatmap = st.tabs(["棒 / 折れ線 / 面 / 積み上げ", "ヒートマップ"])

with tab_line:
    selected_financial = st.multiselect("財務項目（複数選択可）", options=financial_items, default=["Price Discounts",])

    line_df = filtered_df.copy()
    if selected_financial:
        line_df = line_df[line_df["財務項目"].isin(selected_financial)]

    if line_df.empty:
        st.warning("選択条件に一致するデータがありません。条件を変更してください。")
        st.stop()

    base_chart_types = ["折れ線グラフ", "棒グラフ", "面グラフ"]
    chart_type_options = [
        "折れ線グラフ",
        "棒グラフ",
        "面グラフ",
        "積み上げグラフ",
        "100%積み上げグラフ",
    ]

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("##### 軸左")
        l1, l2 = st.columns(2)
        left_versions = l1.multiselect(
            "バージョン（軸左）", options=version_options, default=[default_version], key="left_versions"
        )
        left_y_axis = l2.selectbox("軸ラベル（軸左）", y_axis_options, index=0, key="left_y_axis")
        left_chart_type = st.selectbox(
            "グラフ種類（軸左）",
            chart_type_options,
            index=0,
            key="left_chart_type",
        )
        if left_chart_type == "積み上げグラフ":
            left_label_options = ["なし", "あり", "構成比"]
        else:
            left_label_options = ["あり", "なし"]
        if "left_label_mode" in st.session_state and st.session_state["left_label_mode"] not in left_label_options:
            st.session_state["left_label_mode"] = "あり"
        left_label_mode = st.radio(
            "データラベル（軸左）",
            options=left_label_options,
            index=left_label_options.index("あり"),
            horizontal=True,
            key="left_label_mode",
        )

        stack_by = None
        if left_chart_type in {"積み上げグラフ", "100%積み上げグラフ"}:
            stack_dim = st.selectbox(
                "積み上げ項目",
                options=["チャネル", "カテゴリ", "財務項目"],
                index=0,
                key="line_stack_dim",
            )
            stack_by = resolve_stack_by(
                stack_dim,
                selected_channel_lv1,
                selected_channel_lv2,
                selected_category_lv1,
                selected_category_lv2,
            )

    with rcol:
        st.markdown("##### 軸右")
        r1, r2, r3 = st.columns(3)
        right_versions = r1.multiselect(
            "バージョン（軸右）", options=version_options, default=[default_version], key="right_versions"
        )
        right_y_axis = r2.selectbox("軸ラベル（軸右）", y_axis_options, index=1, key="right_y_axis")
        right_label_mode = r3.radio(
            "データラベル（軸右）", options=["あり", "なし"], index=0, horizontal=True, key="right_label_mode"
        )

        if left_chart_type in {"積み上げグラフ", "100%積み上げグラフ"}:
            right_chart_options = ["折れ線グラフ"]
            right_chart_index = 0
        else:
            right_chart_options = base_chart_types
            right_chart_index = 0
        right_chart_type = st.selectbox(
            "グラフ種類（軸右）",
            right_chart_options,
            index=right_chart_index,
            key="right_chart_type",
            disabled=left_chart_type in {"積み上げグラフ", "100%積み上げグラフ"},
        )

    if left_chart_type in {"積み上げグラフ", "100%積み上げグラフ"}:
        stack_df = line_df.copy()
        if left_versions:
            stack_df = stack_df[stack_df["バージョン"].isin(left_versions)]
        else:
            stack_df = stack_df.iloc[0:0]
        stack_df = stack_df[stack_df["単位"] == left_y_axis]

        if stack_df.empty:
            st.info("積み上げグラフを表示できるデータがありません。選択を見直してください。")
        else:
            grouped_stack = (
                stack_df.groupby(["年月", "年月表示", "バージョン", stack_by], as_index=False)["値"]
                .sum()
                .sort_values(["年月", "バージョン"])
            )
            is_percent_stack = left_chart_type == "100%積み上げグラフ"
            is_ratio_label = left_label_mode == "構成比"
            if is_percent_stack or is_ratio_label:
                grouped_stack["合計"] = grouped_stack.groupby(
                    ["年月", "年月表示", "バージョン"], as_index=False
                )["値"].transform("sum")
            if is_percent_stack:
                grouped_stack["値"] = grouped_stack["値"] / grouped_stack["合計"] * 100
            left_show_labels = left_label_mode in {"あり", "構成比"}
            right_show_labels = right_label_mode == "あり"
            show_stack_labels = left_show_labels and should_show_labels(grouped_stack)
            if is_percent_stack:
                grouped_stack = add_value_labels(grouped_stack, "値", formatter=lambda v: f"{v:.1f}%")
            elif is_ratio_label:
                grouped_stack = grouped_stack.copy()
                ratio = grouped_stack["値"] / grouped_stack["合計"] * 100
                grouped_stack["ラベル"] = ratio.map(lambda v: f"{v:.1f}%")
            else:
                grouped_stack = add_value_labels(grouped_stack, "値")
            grouped_stack["年月×バージョン"] = grouped_stack["年月表示"] + "｜" + grouped_stack["バージョン"]

            version_order = left_versions if left_versions else version_options
            x_order = build_month_version_order(grouped_stack, version_order=version_order)
            grouped_stack["年月×バージョン"] = pd.Categorical(
                grouped_stack["年月×バージョン"],
                categories=x_order,
                ordered=True,
            )

            right_df = (
                line_df[line_df["バージョン"].isin(right_versions) & (line_df["単位"] == right_y_axis)]
                if right_versions
                else pd.DataFrame(columns=line_df.columns)
            )
            right_grouped = (
                right_df.groupby(["年月", "年月表示", "財務項目", "バージョン"], as_index=False)["値"].sum()
                if not right_df.empty
                else pd.DataFrame(columns=["年月", "年月表示", "財務項目", "バージョン", "値"])
            )
            show_right_labels = right_show_labels and should_show_labels(right_grouped)
            right_grouped = add_value_labels(right_grouped, "値") if not right_grouped.empty else right_grouped
            right_grouped["年月×バージョン"] = right_grouped["年月表示"] + "｜" + right_grouped["バージョン"]
            if not right_grouped.empty:
                right_grouped["年月×バージョン"] = pd.Categorical(
                    right_grouped["年月×バージョン"],
                    categories=x_order,
                    ordered=True,
                )

            stack_fig = make_subplots(specs=[[{"secondary_y": True}]])
            for stack_key, g in grouped_stack.groupby(stack_by):
                g = g.sort_values(["年月", "バージョン"])
                stack_fig.add_trace(
                    go.Bar(
                        x=g["年月×バージョン"],
                        y=g["値"],
                        name=f"{stack_key}",
                        text=g["ラベル"] if show_stack_labels else None,
                        texttemplate="%{text}" if show_stack_labels else None,
                        textposition="inside",
                    ),
                    secondary_y=False,
                )

            for (item, ver), g in right_grouped.groupby(["財務項目", "バージョン"]):
                g = g.sort_values(["年月", "バージョン"])
                stack_fig.add_trace(
                    go.Scatter(
                        x=g["年月×バージョン"],
                        y=g["値"],
                        name=f"{item}｜{ver}（軸右）",
                        mode="lines+markers+text" if show_right_labels else "lines+markers",
                        text=g["ラベル"] if show_right_labels else None,
                        textposition="top center",
                    ),
                    secondary_y=True,
                )

            stack_fig.update_layout(
                barmode="stack",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title=legend_title_with_unit(
                    f"{stack_by} / 財務項目 / バージョン", unit="%" if is_percent_stack else "M"
                ),
            )
            stack_fig.update_xaxes(categoryorder="array", categoryarray=x_order)
            if is_percent_stack:
                stack_fig.update_yaxes(title_text="割合(%)", secondary_y=False, range=[0, 100])
            else:
                stack_fig.update_yaxes(title_text=left_y_axis, secondary_y=False)
            stack_fig.update_yaxes(title_text=right_y_axis, secondary_y=True)
            st.plotly_chart(stack_fig, use_container_width=True)
    else:
        left_df = (
            line_df[line_df["バージョン"].isin(left_versions) & (line_df["単位"] == left_y_axis)]
            if left_versions
            else pd.DataFrame(columns=line_df.columns)
        )
        right_df = (
            line_df[line_df["バージョン"].isin(right_versions) & (line_df["単位"] == right_y_axis)]
            if right_versions
            else pd.DataFrame(columns=line_df.columns)
        )

        if left_df.empty and right_df.empty:
            st.warning("選択条件に一致するデータがありません。条件を変更してください。")
        else:
            left_show_labels = left_label_mode == "あり"
            right_show_labels = right_label_mode == "あり"
            fig = build_dual_axis_chart(
                line_df,
                axis_col="年月表示",
                left_cfg={
                    "data": left_df,
                    "chart_type": left_chart_type,
                    "label": left_y_axis,
                    "show_labels": left_show_labels,
                },
                right_cfg={
                    "data": right_df,
                    "chart_type": right_chart_type,
                    "label": right_y_axis,
                    "show_labels": right_show_labels,
                },
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_heatmap:
    hcol1, hcol2, hcol3 = st.columns(3)
    dimension_mode = hcol1.selectbox("Y軸の項目", options=["チャネル", "カテゴリ"], index=0)
    heatmap_value = hcol2.selectbox("値の種類", options=y_axis_options, index=1, key="heatmap_value")
    heatmap_version = hcol3.selectbox(
        "バージョン（ヒートマップ）",
        options=[HEATMAP_FIXED_VERSION],
        index=0,
        key="heatmap_version",
        disabled=True,
    )
    heatmap_fin_items = st.multiselect(
        "財務項目（複数選択可）",
        options=financial_items,
        default=[],
        key="heatmap_fin_items",
    )

    heatmap_df = filtered_df.copy()
    heatmap_df = heatmap_df[heatmap_df["バージョン"] == heatmap_version]
    heatmap_df = heatmap_df[heatmap_df["単位"] == heatmap_value]
    if heatmap_fin_items:
        heatmap_df = heatmap_df[heatmap_df["財務項目"].isin(heatmap_fin_items)]

    if heatmap_df.empty:
        st.info("ヒートマップを表示できるデータがありません。選択を見直してください。")
    else:
        if dimension_mode == "チャネル":
            if selected_channel_lv1 == "-":
                heatmap_y_axis = "channel_lv1"
            elif selected_channel_lv2 == "-":
                heatmap_y_axis = "channel_lv2"
            else:
                heatmap_y_axis = "channel_lv3"
        else:
            if selected_category_lv1 == "-":
                heatmap_y_axis = "category_lv1"
            elif selected_category_lv2 == "-":
                heatmap_y_axis = "category_lv2"
            else:
                heatmap_y_axis = "category_lv3"

        x_order = heatmap_df.sort_values("年月")["年月表示"].dropna().unique().tolist()
        y_order = sorted(heatmap_df[heatmap_y_axis].dropna().unique().tolist())

        pivot = (
            heatmap_df.pivot_table(
                index=heatmap_y_axis,
                columns="年月表示",
                values="値",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=y_order, columns=x_order)
        )

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdBu",
                reversescale=True,
                colorbar_title=legend_title_with_unit(heatmap_value),
            )
        )
        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="年月",
            yaxis_title=heatmap_y_axis,
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
