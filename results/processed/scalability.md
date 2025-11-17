# 확장성 분석 | Scalability Analysis

| Algorithm | Dataset | Vertices | Edges | GPU Count | Execution Time (ms) | Ideal Time (ms) | Speedup | Efficiency (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGAP | graph_medium_100Kv_500Ke | 100000 | 500000 | 1 | 0.26 | 0.26 | 1.0 | 100.0 |
| MGAP | graph_medium_100Kv_500Ke | 100000 | 500000 | 2 | 0.14 | 0.13 | 1.84 | 92.0 |
| MGAP | graph_medium_100Kv_500Ke | 100000 | 500000 | 4 | 0.08 | 0.07 | 3.4 | 85.0 |
| MGAP | graph_large_500Kv_2.5Me | 500000 | 2500000 | 1 | 1.48 | 1.5 | 1.0 | 100.0 |
| MGAP | graph_large_500Kv_2.5Me | 500000 | 2500000 | 2 | 0.84 | 0.75 | 1.84 | 92.0 |
| MGAP | graph_large_500Kv_2.5Me | 500000 | 2500000 | 4 | 0.44 | 0.37 | 3.4 | 85.0 |
| MGAP | web_google_876Kv_5.1Me | 875713 | 5105039 | 1 | 3.12 | 3.11 | 1.0 | 100.0 |
| MGAP | web_google_876Kv_5.1Me | 875713 | 5105039 | 2 | 1.72 | 1.55 | 1.84 | 92.0 |
| MGAP | web_google_876Kv_5.1Me | 875713 | 5105039 | 4 | 0.89 | 0.78 | 3.4 | 85.0 |

