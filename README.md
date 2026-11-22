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

离线是默认路径。可用 `OPENAI_API_KEY`、`OPENAI_MODEL`、可选 `OPENAI_BASE_URL` 加 `--online` 复跑。每次 CLI 运行共享请求与 token 预算，默认上限分别为 60 与 250,000，SDK 自动重试关闭。用量未知会停止继续调用，预算中断的任务保留 Trace 并排除出配对汇总。下方命令可复跑一次限额内的在线配对。线上失败或用量未知时保留失败记录。

## Mode B 模型规划

`investigate --online` 先执行一次共同的整体对比，后续由模型决定动作。它可以分组、在同一两期人群上条件下钻、核查一个人群总量，或停止。模型给出动作或停止决定、调查疑问和 Evidence ID。方向预测与预案均可选。SQL 编译器固定指标和窗口，工具从结果中确定最显著的分组。空白、JSON 解析失败或重复动作至多使用一个剩余决策名额纠正。离线路径继续使用 Rule-Adaptive，`auto` 的显式单维分解走固定 Workflow。

```bash
python -m data_analysis_agent eval --mode-b --budget 5 --bike --output evaluation/replay_offline.json
MAX_LLM_REQUESTS=8 MAX_LLM_TOTAL_TOKENS=100000 python -m data_analysis_agent eval --mode-b --online --case reliability_localized --budget 5 --output evaluation/replay_online.json
```

3、5、7 是各自的真实动作上限，整体对比计 1 次。LLM-Frozen 和 LLM-Adaptive 共用首条证据、模型、工具与预算。Frozen 后续只看到自己的历史决策，最后由同一确定性渲染器汇总实际工具结果。一次可计分的模型重规划要求模型在观察前留下下一步，工具随后反驳待验证假设，模型再引用新证据执行相关的替代动作。

### 真实模型可靠性实验

[开发诊断、冻结对照和完整 Trace](evaluation/mode_b_reliability.json)。旧记录确认过超时、JSON 解析失败、重复动作，以及把 SP 州过滤值混入星期分组的情况。旧日志缺少响应长度和结束原因，无法追溯每个解析失败的具体成因。本轮在开发集复现两次 `finish_reason=length`，其中一次可见内容为空。采用短协议和 3,072 token 上限后，开发集拿到了完整 JSON。请求要求关闭思考，网关仍返回了推理字段；用量和输出长度均以实测记录为准。[DeepSeek JSON 模式](https://api-docs.deepseek.com/guides/json_mode/) · [DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)

采用 85 秒单次等待后，开发集的两个模型组都完成两步调查，4/4 响应为合法 JSON。Adaptive 的 E1 测得晚到率从 16.00% 变为 18.14%；E2 分解星期变化，模型引用 E2 后改查州分解，E3 定位 MG 贡献 +2.14 个百分点，最后在分析预算用尽时交付报告。这是依据新证据选择下一步的真实记录。

冻结后运行两道未用于调试的 Olist 题，分析预算分别为 5 次和 3 次。每题四组共享快照和动作上限。

| 策略 | 正确 | 调查完成 | 可解析 JSON | SQL | 请求 | Token | 两题总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Workflow | 1/2 | 2/2 | — | 16 | 0 | 0 | 0.39 s |
| Rule-Adaptive | 2/2 | 2/2 | — | 16 | 0 | 0 | 0.38 s |
| LLM-Frozen | 0/2 | 1/2 | 6/6 | 14 | 6 | 16,579 | 61.77 s |
| LLM-Adaptive | 2/2 | 2/2 | 5/6 | 14 | 6 | 19,681 | 53.25 s |

两模型组提出的动作均通过校验（各 5/5）。Frozen 一题主动报告证据不足，另一题交付了报告但漏掉机制。Adaptive 的数字和 4/4 条统计声明均可回溯。局部变化用例的一次非 JSON 响应长 186 字符，结束原因为 `stop`；占用一个决策名额纠正后，模型完成了调查。E2 反驳 SP 假设，模型引用 E2 查州分解，E3 定位 MG，再由 E4 定位到 MG 的周四。模型没有在观察前提交可比较的预案，严格反证重规划仍为 0 次。Adaptive 在这两题上胜过固定顺序和 Frozen，准确率与规则策略持平，耗时明显更长。两题不足以估计一般性优势。

含调试共发出 20 次请求，已计量 50,372 token，另有 1 次超时的用量未知。SDK 自动重试关闭，达到请求上限后停止。

测试按 SQL 与语义、调查、分析、数据管道、接口、评测和工作流合并为 7 个文件、260 个用例。保留了安全边界、关键回归及线上协议测试，生产模块和依赖保持不变。

### 上轮消融

[原逐案结果、在线请求与 Trace](evaluation/mode_b_followup.json)。上轮新增两个冻结用例。平衡单元格的四处干预在整体及单维边际上抵消，Top-3 至少覆盖其中三个才算完成；双变化代理用例同时包含星期效应和由订单状态变化产生的代理信号。

| 动作预算 | 固定 Workflow | Rule-Adaptive | 两组各自的 SQL 调用 |
| --- | ---: | ---: | ---: |
| 3 | 2/7 | 5/7 | 36 |
| 5 | 6/7 | 6/7 | 56 |
| 7 | 6/7 | 6/7 | 76 |

5 次预算下 Rule-Adaptive 有 3 次有效重规划，固定 Workflow 为 0 次。7 次预算下两组都遗漏了双变化用例的第二个机制，重叠的状态代理贡献占了两个排名位置。上述离线对照共用 Olist 快照，模型请求均为 0。

真实模型使用本机配置的 `deepseek-v4.1-flash`。正式配对完成了历史单维任务（3 次）和条件效应任务（5 次）。固定 Workflow 与 Rule-Adaptive 都是 2/2，LLM-Frozen 和 LLM-Adaptive 都是 0/2。双变化代理任务因超时且用量未知，保留逐案记录，排除出配对汇总。所有在线阶段共 25 次请求，已知用量 87,267 token，另有 2 次请求用量未知。

条件效应任务确实出现了模型动作调整。LLM-Adaptive 原定全量按星期分组，看到 E2 的 SP 贡献后改查 SP 子群的星期分解。E3 已实际执行，随后模型输出无法解析，调查未完成。由于前一个假设得到支持，这次调整在反证口径下计为 0 次有效重规划。LLM-Frozen 未查到州与星期的联合单元。现有在线样本没有支持 LLM 相对确定性基线的质量或成本优势，7 次预算未做在线配对。历史单维任务的首轮试跑中，Workflow 与在线 Adaptive 都完成任务，路由到 Workflow 节省了 2 次模型请求；本例只体现显式单维任务的成本。

Mock 回归单独记录。以真实 Olist 副本上的受控干预为底表，E1 显示晚到率从 9.37% 升至 14.06%；H1 预测 SP 州内变化不大，预定下一步按订单状态分组。E2 测得 SP 州内贡献约 +4.69 个百分点并反驳 H1。脚本模型引用 E2，实际改为在 SP 内按星期下钻。E3 测得周三集中贡献，并触发有证据的停止。这条 Trace 验证协议和执行，属于 Mock。

## V3.2 原始冻结实验

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
