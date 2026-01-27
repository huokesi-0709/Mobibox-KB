# MoniBox 评分系统（Scoring & Re-ranking）

本目录用于管理“测试评分如何影响检索排序”的策略，使闭环成立：
- 测试人员可对片段打分 / 启用 / 停用
- 下一次检索立即生效（不需要重建向量库）

## 文件说明
- policy.json  
  重排策略配置。当前 preset=medium（中等强度）。

## 重排公式（默认）
distance 越小越相关，最终重排距离：

d_final = d - w_quality * (quality_score / 5) - w_enabled * I(status == 启用)

排序按 d_final 升序（越小越靠前）

## 运行方式
1) 查询（带重排）
python -m scripts.query_demo --q "我好害怕，喘不过气" --topk 5

2) 打分/启用/停用
python -m scripts.rate_chunk --display_id "<显示ID>" --score 5 --status 启用
python -m scripts.rate_chunk --display_id "<显示ID>" --status 停用

3) 再查询（立刻看到排序变化）
python -m scripts.query_demo --q "我好害怕，喘不过气" --topk 5