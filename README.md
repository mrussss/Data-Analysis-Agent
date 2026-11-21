# Data Analysis Agent

面向结构化业务数据的可信问数和证据驱动调查。CLI 和 Streamlit 共用指标、只读 SQL 和路由。开发契约见 [V3.2](Data_Analysis_Agent_Project_Spec_v3_2.md)。

## 最短复现

在 WSL2 的本仓库根目录使用 Python 3.11+ 和 `curl`。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m scripts.prepare_data all

python -m data_analysis_agent query 'gmv from 2018-03-01 to 2018-07-01 where status=delivered by month'
python -m data_analysis_agent investigate 'why late_delivery_rate from 2018-05-01 to 2018-06-01 versus 2018-04-01 to 2018-05-01'
python -m data_analysis_agent eval --bike --output evaluation/replay.json
python -m pytest
```

`auto` 自动选择问数、单维固定分解或开放调查。调查可用 `--budget 3` 限制分析动作，或用 `--baseline-db PATH` 比较两个只读快照。复跑受控反例时执行 `python -m data_analysis_agent eval --case conditional`。开发集用 `eval --split dev`，包含 1 道问数、1 道调查。运行 `streamlit run app.py` 可在本地使用相同的离线链路。

问数接受 `<metric> from <start> to <end> [where field=value [and field=value]] [by day|month]`。日期左闭右开。Olist 支持订单数、三种金额口径（商品成交额、含运费成交额和支付额）、每单平均商品数、平均配送时长、晚到率。过滤字段为 `state`、`weekday`（周一为 1）、`status`。晚到按实际送达时间与预估送达时间比较，只纳入两个时间都有记录的订单。Bike 支持 `rentals`、`average_hourly_rentals`，可按 `workingday`、`weather` 或 `season` 调查。超出支持范围的问数会明确拒答。

离线是默认路径。可用 `OPENAI_API_KEY`、`OPENAI_MODEL`、可选 `OPENAI_BASE_URL` 加 `--online` 复跑。每次 CLI 运行共享请求与 token 预算，默认上限分别为 60 与 250,000，SDK 自动重试关闭。用量未知会停止继续调用，预算中断的任务保留 Trace 并排除出配对汇总。建议从 `python -m data_analysis_agent eval --online --case conditional --output evaluation/online_conditional.json` 开始。本次无模型凭据，在线实验未运行。

## 冻结实验

[机器可读结果与 Trace](evaluation/v3_2_offline.json)保存快照和题库哈希、独立真值、逐任务结果、调用数和延迟。Olist V2 的 9 份官方 CSV 均通过 SHA-256 校验。原始数据和干预库留在本地。UCI 按官方压缩包里的 `hour.csv` 计算，共 17,379 条小时记录。数据分别遵循 Olist CC BY-NC-SA 4.0 与 UCI CC BY 4.0。[Kaggle Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) · [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset)

冻结集含 10 道问数、1 道固定调查和 8 道开放调查。离线问数使用同一组预写 SQL 提案，其中含刻意植入的语义错误，数字由独立 SQL 评分。完整调查用确定性统计策略，真实模型请求与 token 均为 0。

| 冻结对照 | 正确 | SQL 调用 | P50 / P95（ms） |
| --- | ---: | ---: | ---: |
| 裸问数（仅问数） | 3/10 | 11 | 19.52 / 26.77 |
| QueryContract（仅问数） | 10/10 | 8 | 20.23 / 34.69 |
| A-only（混合任务） | 11/19 | 26 | 34.41 / 41.72 |
| 固定 Workflow（混合任务） | 19/19 | 112 | 34.21 / 374.66 |
| 完整 B 策略（混合任务） | 19/19 | 112 | 34.43 / 380.42 |
| 自适应路由（混合任务） | 19/19 | 112 | 34.91 / 403.65 |
| 完整确定性扫描（仅调查） | 9/9 | 104 | 318.03 / 346.54 |
| 移除证据重规划（仅调查） | 9/9 | 104 | 323.00 / 439.79 |

- **问数**。契约拦下 7 个原本会接受的错误答案，2 道应拒问题均正确拒答。裸问数在商品与支付明细同时连接时重复计价，在修复实例里丢掉时间与状态过滤。时间边界提案恰好与金标数值相同，左闭右开的行为另由边界测试验证。
- **调查**。开放任务统一限制 3 个分析动作时，完整 B 为 7/8，扫描、固定 Workflow、移除重规划均为 4/8。预算放到 5 个动作时均为 7/8，完整预算均为 8/8。完整 B 的 4 次观测到的反证机会均改变后续动作，统计声明 16/16 可回溯，两个对照均无虚假归因。5 个注入案例的机制命中 Top-1 为 3/5，Top-3 为 5/5，抵消案例的两个方向都进入 Top-3。构成变化和关联代理仍可抢占首位，注入调查中有 3 条有数据依据、但偏离主要机制的统计归因。
- **路由**。混合任务中自动路由相对 A-only 提升完成率并增加工具成本。它与完整 B、固定 Workflow 的全预算正确率和调用数持平。各策略共用问数链路，唯一一道显式单维调查对各策略开放同一维度，无法凭此估计路由节省。短预算结果是前缀消融，小样本下延迟分位数和效果估计均不稳定。
- **跨域**。UCI 的 2012 年 6 月工作日租车量为 138,257，6 月至 7 月每记录小时均值由 281.71 降至 273.67。两项均以 CSV 独立复核，调查沿用相同状态和分解。

## 三条链路

| 链路 | 入口与关键代码 |
| --- | --- |
| 可信问数 | [QueryContract、编译与单次修复](src/data_analysis_agent/query_contract.py) → [只读执行和超时](src/data_analysis_agent/sql_executor.py) → [结果校验](src/data_analysis_agent/result_validation.py) |
| 证据调查 | [measure、assess、investigate](src/data_analysis_agent/investigation.py) 使用共同的口径，保留样本量、分解数字、证据 ID、假设状态和下一步。反证发生时记录观察前预选动作和实际执行动作。 |
| 路由与消融 | [run_task](src/data_analysis_agent/routing.py) → [同快照策略对照与评分](src/data_analysis_agent/benchmark.py)，[注入和独立真值](src/data_analysis_agent/benchmark_cases.py)仅供评测器使用。 |

取舍与面试追问

- 怎样让 SQL 修复无法扩大问题范围？日期、过滤、粒度与指标先冻结，执行层只接受该契约生成的两种 SQL 形式。任意等价 SQL 的判断留在编译器支持范围之外，代价是通用性。
- 怎样判断反证真的改变了行动？对照观察前预选动作，检查新证据是否推翻原假设以及下一动作是否针对被反证的维度。全扫描同样能找到结果时，等预算前缀可以衡量先后顺序的价值。
- 统计分解成立后还缺什么？总量分解和证据引用可以复核，真实业务因果仍需要额外设计或数据。受控副本的编辑清单为注入样本提供真值。

## 复用与新增

- **来自上游**。保留 `5456465/data-analysis-agent` 的 Git 历史和 MIT 许可，基线为 `0784efcfa1e3aa3b5ee8cb9efbf158db501427d9`。复用 Olist 类型化导入和订单预聚合视图、语义词典、SQL 生成和修复（裸问数对照）、只读执行、结果校验、Trace、Streamlit 页面入口。[主底座](https://github.com/5456465/data-analysis-agent)
- **本次新增**。冻结语义的查询编译和有限修复、SQL 执行超时、证据反馈的调查状态与反证后重规划、双模式路由、独立注入和消融评分、Bike 轻量映射。KPI Root-Cause Agent 提供统计钻取、注入和强基线的设计参考，参考版本为 `f85567b209879aa7cbfdd2db3622c9e148dfe671`。[方法参考](https://github.com/nivi9944/kpi-rca-agent)
