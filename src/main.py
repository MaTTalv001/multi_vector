import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO

# superlinked 関連
from superlinked import framework as sl
from superlinked.evaluation.vector_sampler import VectorSampler

# クラスタリング & 次元削減
from sklearn.cluster import HDBSCAN
import umap

st.title("WBS Clustering App (superlinked)")

# CSVファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    # データフレームに読み込み
    df = pd.read_csv(uploaded_file)
    st.write("アップロードしたデータ:")
    st.dataframe(df.head(10))  # 先頭10行程度を表示

    # superlinked は id 列が必要
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    
    # 埋め込み対象カラムの複数選択
    candidate_cols = [c for c in df.columns if c != "id"]
    selected_cols = st.multiselect(
        "ベクトル化(埋め込み)するカラムを選択してください",
        candidate_cols,
        default=["WBS_CODE_lv01", "WBS_CODE_lv02", "WBS_CODE_lv03", "Activity"]
    )
    
    # カラム重みのスライダー
    col_weights = {}
    for col in selected_cols:
        col_weights[col] = st.slider(
            f"カラム [{col}] の重み",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
    
    if st.button("分析を実行"):
        if not selected_cols:
            st.warning("埋め込み対象のカラムが選択されていません。")
            st.stop()

        st.write("クラスタリングを実行します...（少しお待ちください）")
        
        # ------------------------------------------------
        # 1. 動的に superlinked.Schema を作成
        # ------------------------------------------------
        annotations_dict = {"id": sl.IdField}
        for col in selected_cols:
            annotations_dict[col] = sl.String
        
        class_attrs = {
            "__annotations__": annotations_dict
        }
        DynamicSchema = type("DynamicSchema", (sl.Schema,), class_attrs)
        wbs_schema = DynamicSchema()
        
        # ------------------------------------------------
        # 2. TextSimilaritySpace で埋め込み設定
        # ------------------------------------------------
        spaces = []
        for col in selected_cols:
            space = sl.TextSimilaritySpace(
                text=getattr(wbs_schema, col),
                model="sentence-transformers/all-mpnet-base-v2"
            )
            spaces.append(space)
        
        wbs_index = sl.Index(spaces=spaces)
        
        # ------------------------------------------------
        # 3. InMemoryExecutor / App を作成する関数
        #    こちらは @st.cache_resource にする
        # ------------------------------------------------
        @st.cache_resource
        def create_app(df: pd.DataFrame, _schema, _index):
            dataframe_parser = sl.DataFrameParser(schema=_schema, mapping={})
            source = sl.InMemorySource(_schema, parser=dataframe_parser)
            executor = sl.InMemoryExecutor(sources=[source], indices=[_index])
            app = executor.run()
            # CSVのデータを put
            source.put([df])
            return app

        # resource cache に格納 (InMemoryApp はピクル化できないが resource ならOK)
        app = create_app(df, wbs_schema, wbs_index)

        # ------------------------------------------------
        # 4. ベクトル取得 & 重み付け
        # ------------------------------------------------
        vs = VectorSampler(app=app)
        vector_collection = vs.get_all_vectors(wbs_index, wbs_schema)
        
        vectors = vector_collection.vectors  # numpy配列
        id_list = [int(i) for i in vector_collection.id_list]

        # カラム(=spaces)毎に 768次元ずつ重みを適用
        base_dim = 768
        start_dim = 0
        for col, space in zip(selected_cols, spaces):
            end_dim = start_dim + base_dim
            weight = col_weights[col]
            vectors[:, start_dim:end_dim] *= weight
            start_dim = end_dim
        
        # ベクトルDataFrame
        vector_df = pd.DataFrame(vectors, index=id_list)
        
        # ------------------------------------------------
        # 5. HDBSCANクラスタリング
        # ------------------------------------------------
        hdbscan_model = HDBSCAN(min_cluster_size=2, metric="cosine")
        hdbscan_model.fit(vector_df.values)
        
        labels = hdbscan_model.labels_
        label_df = pd.DataFrame(labels, index=vector_df.index, columns=["cluster_label"])
        st.write("クラスタ数 (ラベルごとの件数):")
        st.write(label_df["cluster_label"].value_counts())

        # 元のデータフレームにクラスタラベルを追加
        df_with_clusters = df.copy()
        df_with_clusters['cluster_label'] = label_df['cluster_label']
        
        # クラスタラベル付きのデータフレームを表示
        st.write("クラスタリング結果を付与したデータ:")
        st.dataframe(df_with_clusters)
        
        # ------------------------------------------------
        # 6. UMAP で次元削減 → Altair可視化
        # ------------------------------------------------
        umap_model = umap.UMAP(random_state=0, transform_seed=0, n_jobs=1, metric="cosine")
        umap_vectors = umap_model.fit_transform(vector_df.values)
        
        umap_df = pd.DataFrame(umap_vectors, columns=["dimension_1", "dimension_2"], index=vector_df.index)
        umap_df = umap_df.join(label_df)
        
        chart = (
            alt.Chart(umap_df.reset_index())
            .mark_circle(size=64)
            .encode(
                x="dimension_1",
                y="dimension_2",
                color="cluster_label:N",
                tooltip=["index", "cluster_label"]
            )
            .properties(
                width=600,
                height=500,
                title="UMAPの2次元空間 (HDBSCANクラスタラベルで色分け)"
            )
            .configure_title(fontSize=16, anchor="middle")
            .configure_legend(
                strokeColor="black",
                padding=10,
                cornerRadius=10,
                labelFontSize=14,
                titleFontSize=14
            )
            .configure_axis(titleFontSize=14, labelFontSize=12)
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.success("分析が完了しました！")

else:
    st.info("左側のサイドバーか上部エリアから CSV ファイルをアップロードしてください。")
# サイドバーにキャッシュ制御用のセクションを追加
with st.sidebar:
    st.write("### キャッシュ制御")
    cache_options = st.multiselect(
        "初期化するキャッシュの種類を選択:",
        ["Resource Cache", "Data Cache"],
        default=["Resource Cache", "Data Cache"]
    )
    
    if st.button("選択したキャッシュを初期化"):
        if "Resource Cache" in cache_options:
            st.cache_resource.clear()
        if "Data Cache" in cache_options:
            st.cache_data.clear()
        st.success(f"選択されたキャッシュ ({', '.join(cache_options)}) を初期化しました")