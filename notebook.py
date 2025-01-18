import os
from io import StringIO
import sys
import altair as alt
import pandas as pd
from superlinked import framework as sl
from superlinked.evaluation.vector_sampler import VectorSampler

alt.renderers.enable(sl.get_altair_renderer())
alt.data_transformers.disable_max_rows()
pd.set_option("display.max_colwidth", 190)

# サンプルデータ（ヘッダー含め全20行）
data = """WBS_CODE_lv01,WBS_CODE_lv02,WBS_CODE_lv03,Activity,Department
A-00011,臨床開発,リクルーティング,A-00011の被験者登録,臨床開発部
A-00012,臨床開発,モニタリング,A-00012のモニタリング実施,臨床開発部
A-00013,基礎研究,データ整理,A-00013の研究データ整理,研究部
A-00014,基礎研究,実験計画,A-00014のマウス実験準備,研究部
A-00015,製造,ライン拡張,A-00015の新ライン立ち上げ,製造部
A-00016,製造,設備点検,A-00016の設備保守点検,製造部
A-00017,品質管理,検体検査,A-00017の検体スクリーニング,品質管理部
A-00018,品質管理,規格策定,A-00018の品質基準見直し,品質管理部
A-00019,薬事,承認申請,A-00019の承認書類作成,薬事部
A-00020,薬事,照会対応,A-00020の当局照会対応,薬事部
A-00021,臨床開発,試験実施,A-00021の臨床試験実施,臨床開発部
A-00022,臨床開発,試験実施,A-00022の臨床試験予備調査,臨床開発部
A-00023,基礎研究,文献調査,A-00023の新規化合物文献調査,研究部
A-00024,基礎研究,試験デザイン,A-00024の実験プロトコル策定,研究部
A-00025,製造,生産管理,A-00025の生産スケジュール立案,製造部
A-00026,品質管理,試験法開発,A-00026のHPLC法検証,品質管理部
A-00027,薬事,報告書作成,A-00027の国内報告書ドラフト,薬事部
A-00028,臨床開発,報告書作成,A-00028の試験結果報告書作成,臨床開発部
A-00029,製造,原料購買,A-00029の原材料購買計画,製造部
A-00030,品質管理,手順書更新,A-00030のSOP改訂,品質管理部
"""

# pandasで読み込む
df = pd.read_csv(StringIO(data))

# superinkedはインデックスが必要
df = df.reset_index().rename(columns={"index": "id"})

# スキーマの設定（カラム分それぞれ設定）
class WBSSchema(sl.Schema):
    WBS_CODE_lv01: sl.String
    WBS_CODE_lv02: sl.String
    WBS_CODE_lv03: sl.String
    Activity: sl.String
    id: sl.IdField

# ここでインスタンス化
wbs = WBSSchema()

# textual characteristics are embedded using a sentence-transformers model
lv01_space = sl.TextSimilaritySpace(text=wbs.WBS_CODE_lv01, model="sentence-transformers/all-mpnet-base-v2")
lv02_space = sl.TextSimilaritySpace(text=wbs.WBS_CODE_lv02, model="sentence-transformers/all-mpnet-base-v2")
lv03_space = sl.TextSimilaritySpace(text=wbs.WBS_CODE_lv03, model="sentence-transformers/all-mpnet-base-v2")
Activity_space = sl.TextSimilaritySpace(text=wbs.Activity, model="sentence-transformers/all-mpnet-base-v2")

# インデックスの設定
wbs_index = sl.Index(spaces=[lv01_space, lv02_space, lv03_space, Activity_space])

# ここもしかしたら全部のカラムいるのかな？何しているかよく分からず
dataframe_parser = sl.DataFrameParser(
    schema=wbs,
    mapping={wbs.WBS_CODE_lv01: "WBS_CODE_lv01"},
)

# インメモリで処理
source: sl.InMemorySource = sl.InMemorySource(wbs, parser=dataframe_parser)
executor: sl.InMemoryExecutor = sl.InMemoryExecutor(sources=[source], indices=[wbs_index])
app: sl.InMemoryApp = executor.run()
# 実行
source.put([df])

# ベクトル化
vs = VectorSampler(app=app)
vector_collection = vs.get_all_vectors(wbs_index, wbs)
vectors = vector_collection.vectors
vector_df = pd.DataFrame(vectors, index=[int(id_) for id_ in vector_collection.id_list])
vector_df

# 次元削減とクラスタリング
from sklearn.cluster import HDBSCAN
import umap

hdbscan = HDBSCAN(min_cluster_size=2, metric="cosine")
hdbscan.fit(vector_df.values)

label_df = pd.DataFrame(
    hdbscan.labels_, index=vector_df.index, columns=["cluster_label"]
)
label_df["cluster_label"].value_counts()

umap_transform = umap.UMAP(random_state=0, transform_seed=0, n_jobs=1, metric="cosine")
umap_transform = umap_transform.fit(vector_df)
umap_vectors = umap_transform.transform(vector_df)
umap_df = pd.DataFrame(
    umap_vectors, columns=["dimension_1", "dimension_2"], index=vector_df.index
)
umap_df = umap_df.join(label_df)

alt.Chart(umap_df).mark_circle(size=64).encode(
    x="dimension_1", y="dimension_2", color="cluster_label:N"
).properties(
    width=600, height=500, title="UMAP Transformed vectors coloured by cluster labels"
).configure_title(
    fontSize=16,
    anchor="middle",
).configure_legend(
    strokeColor="black",
    padding=10,
    cornerRadius=10,
    labelFontSize=14,
    titleFontSize=14,
).configure_axis(
    titleFontSize=14, labelFontSize=12
)


#  もし、特定のカラムのベクトルのウェイトに傾斜をかけた場合のベクトル合成
vector_df_copy = vector_df.copy()
# 0列目～767列目をスライスして2倍にする
vector_df_copy.iloc[:, 0:768] = vector_df_copy.iloc[:, 0:768] * 2

print("\n【更新後のdf】")
vector_df_copy.head()

hdbscan = HDBSCAN(min_cluster_size=2, metric="cosine")
hdbscan.fit(vector_df_copy.values)

label_df = pd.DataFrame(
    hdbscan.labels_, index=vector_df_copy.index, columns=["cluster_label"]
)
label_df["cluster_label"].value_counts()

umap_transform = umap.UMAP(random_state=0, transform_seed=0, n_jobs=1, metric="cosine")
umap_transform = umap_transform.fit(vector_df_copy)
umap_vectors = umap_transform.transform(vector_df_copy)
umap_df = pd.DataFrame(
    umap_vectors, columns=["dimension_1", "dimension_2"], index=vector_df_copy.index
)
umap_df = umap_df.join(label_df)

alt.Chart(umap_df).mark_circle(size=64).encode(
    x="dimension_1", y="dimension_2", color="cluster_label:N"
).properties(
    width=600, height=500, title="UMAP Transformed vectors coloured by cluster labels"
).configure_title(
    fontSize=16,
    anchor="middle",
).configure_legend(
    strokeColor="black",
    padding=10,
    cornerRadius=10,
    labelFontSize=14,
    titleFontSize=14,
).configure_axis(
    titleFontSize=14, labelFontSize=12
)