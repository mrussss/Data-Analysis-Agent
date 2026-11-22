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

## Mode B 业务调查

旧实验主要在同一晚到率口径下切分州、星期和订单状态。多数题可以用几次预写 SQL 解决，受控题则集中在少数配送日期或状态编辑。原评分对注入定位和统计对账很有用，但很少测试跨表佐证、竞争解释、事件主体和时间，两个真实冻结题也无法估计通用收益。

### 数据与题目

| 类型 | 冻结题 | 数据与内容 |
| --- | ---: | --- |
| A 常规统计 | 8 | 原始 Olist。订单、GMV、运费和商品明细量的贡献定位，另有晚到率的构成／组内分解 |
| B 跨指标、跨表 | 8 | 原始 Olist。消费金额与购买量分离、订单篮子变化、售出商品和卖家广度、评价覆盖、晚到与差评分层 |
| C 外部上下文 | 8 | 独立虚构面板。召回范围、优惠到期、区域渠道、市场休市、竞争品进入、补货、流量转化、产品更新 |
| D 误导与证据不足 | 6 | 同期负面报道叠加缺货、相似主体、迟发事件、多事件共存、库存缺测和指标稳定 |

另有四道独立开发题，每类一道。冻结题与旧冻结题分开。划分、问题、答案和正式在线顺序在首次真实请求前确定。测试集已经公开在仓库中，适用于本轮模型提示隔离和复跑，后续调过这批题的实验需要新的留出集。

A/B 使用现有 DuckDB 的真实商业观测，保留原始记录。Olist 含 99,441 条订单、112,650 条商品明细、99,224 条评价和 3,095 个卖家。商品分类沿用官方译名。GMV 是商品标价之和，排除运费。类别评价先对订单去重，再按评价回答时间、创建时间和稳定键选取最新一条。配送和差评的分母各自取有对应字段的订单。类别订单在不同类别间可以重叠。Olist 没有可靠的品牌、库存、流量和优惠台账。M5 的商品销量层级和价格、Favorita 的品类销量、促销和节假日适合其他问题，本轮采用现成 Olist 加小面板即可复现主体错配、时序倒置和库存缺测。 [Olist 与许可](docs/DATASET.md) · [M5 主办方论文](https://doi.org/10.1016/j.ijforecast.2021.11.013) · [Favorita 官方数据说明](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

C/D 每题按源码生成 28 日、168 行的 SKU／区域／对照面板及对应的本地事件记录，全部主体、事件、数据和实验机制均为虚构。每道题的库存、销量、价格等改动只进入生成器，事件档案只给事件 ID、主体、日期、范围、原文和来源。真实 Olist 表与虚构面板互不写入。事件先按日期和主体名检索，保留相似主体用于核查，报告时再验证精确主体与时序。日期以已查到的持续异常起点为准，否则使用观察窗起点。匹配的时间容差为起点前 3 日至起点当天。重合只支撑候选解释。D 的缺货实验通过库存状态和受影响 SKU 的销量核对受控机制。

### 调查协议与评分

四组共享同一快照、目录、只读 SQL、事件档案、初始比较和最多五次证据动作。首个比较占一次。两期比例分解复用现有统计器，分别执行两条 SQL，原始 SQL 调用数另外记录。Fixed Workflow 在看新增证据前按公开问题预排动作。Rule-Adaptive 用相同的初始选择规则，再根据事件范围或日期执行有针对性的检查。LLM-Frozen 只看初始比较后提交完整动作列表。LLM-Adaptive 逐步看证据并选择动作或停止。两组都可在规划结束后读已取得的证据并提交同样格式的报告。

评分要求结论状态正确、覆盖题目要求的事实，且每条事实由所引 Evidence ID 支持。事件检索、主体／时间匹配、受控机制、竞争证据、错误归因、正确保留归因、预算内完成和反证后的动作变化分别记录。成功的空检索可回答证据不足题。未经充分检查的候选解释与断言错误机制分别计数。方向判断使用 8% 相对差或 2.5 个百分点的比例差。阈值代表本题的实质变化，统计置信区间另需研究设计。观察题评分覆盖可计算的事实和校准结论。完整的金标、执行 SQL、原始证据、动作顺序、给规划器可见的证据 ID、模型用量诊断和协议哈希随 Trace 保存。

### 冻结集离线结果

| 类型 | 策略 | 正确 | 必答事实 | 声明有据 | SQL＋事件检索 | 平均耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Fixed Workflow | 8/8 | 9/9 | 46/46 | 20＋0 | 83.19 ms |
| A | Rule-Adaptive | 8/8 | 9/9 | 46/46 | 20＋0 | 81.03 ms |
| B | Fixed Workflow | 8/8 | 26/26 | 82/82 | 20＋0 | 100.72 ms |
| B | Rule-Adaptive | 8/8 | 26/26 | 82/82 | 20＋0 | 102.75 ms |
| C | Fixed Workflow | 8/8 | 32/32 | 72/72 | 28＋8 | 43.85 ms |
| C | Rule-Adaptive | 8/8 | 32/32 | 72/72 | 28＋8 | 44.54 ms |
| D | Fixed Workflow | 6/6 | 20/20 | 49/49 | 15＋6 | 29.67 ms |
| D | Rule-Adaptive | 6/6 | 20/20 | 49/49 | 15＋6 | 32.12 ms |

两组模型调用与 token 成本均为零，30/30 在动作预算内交付报告。两组事件检索和主体／时间匹配均为 C 8/8、D 6/6，D 正确保留或拒绝归因 5/5，缺货机制识别 1/1。两组均为 30/30。召回题最初只接受按 SKU 查库存，离线审计确认全店两期均全量覆盖且 100% 有货时，已观测 SKU 同样可排除缺货。评分增加等价证据校验，题面、数据和其他评分要求维持冻结。 [冻结集评分与逐案 Trace](evaluation/business_offline.json) · [开发集离线 Trace](evaluation/business_offline_dev.json)

召回题的 Rule-Adaptive Trace 中，E1 发现两期销量由 2,800 降至 2,016。E2 找到仅针对 core 的虚构召回记录，E3 将销量缺口定位到 core，E4 将预排的全店库存查询替换为 core 库存查询，测得可用率仍为 100%，E5 核查投诉率上升。这条记录展示了外部证据如何改变实际 SQL 调查。证据支持保留召回作为候选解释。

### 真实模型试跑与续跑

| 阶段 | 策略 | 样本 | 完成 | 请求 | 已计量 Token／美元 | 延迟 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 开发集 C | LLM-Frozen | 1 次尝试 | 0/1 | 1 | 未返回／未知 | 335.95 ms |
| 开发集 C | LLM-Adaptive | 0 | — | 0 | — | — |
| 冻结集 A–D | LLM-Frozen | 0 | — | 0 | — | — |
| 冻结集 A–D | LLM-Adaptive | 0 | — | 0 | — | — |

配置的 OpenAI 兼容网关在首个真实请求上返回 `APIConnectionError`，没有 HTTP 状态、模型输出或用量。当前环境也未解析到该网关域名。SDK 自动重试关闭，未知用量触发停止。新任务没有真实的跨来源模型 Trace 或任何可计分的在线配对。旧实验的真实 Trace 使用 SQL 证据，仍可在旧报告复查。 [本轮在线失败 Trace](evaluation/business_online_dev.json) · [旧在线 Trace](evaluation/mode_b_reliability.json)

现有证据支持在结构化数据、较小动作空间和清楚的诊断信号下优先使用确定性策略。Rule-Adaptive 的路径变化来自事件范围触发条件下钻，Fixed Workflow 的全量库存证据也足以回答本题。LLM 的增量价值仍待验证。当前事件档案很小，合成面板噪声低，Olist 月窗口长短不完全一致，B 类问题多给出检查方向，公开题库可能被后续模型或提示熟悉。后续实验优先复跑预选配对，再用更少提示和更加含混的候选证据测试选工具的收益。

```bash
python -m scripts.prepare_data olist
python -m data_analysis_agent eval --business --output evaluation/business_offline.json
python -m data_analysis_agent eval --business --split dev --output evaluation/business_offline_dev.json
python -m data_analysis_agent investigate --business --entity health_beauty \
  --before 2018-03-01 2018-04-01 --current 2018-04-01 2018-05-01 --budget 5 \
  'Compare complaints with late delivery and check late and on-time groups.'
python -m pytest
```

网关恢复后，新的一轮先跑开发集，再把实际用量扣除后运行预定冻结配对。每轮最多 40 请求和 160,000 已计量 token。未知用量会阻止原轮续跑，需先用网关用量记录核对并保留原失败 Trace。可以把已核对的历史账本通过多个 `--prior-results` 累计扣除。费用只在提供方明确返回 USD 时记录。

```bash
python -m data_analysis_agent eval --business --split dev --online \
  --output evaluation/business_online_dev_replay.json
python -m data_analysis_agent eval --business --online \
  --prior-results evaluation/business_online_dev_replay.json \
  --output evaluation/business_online_test_replay.json
```

[工具及 SQL](src/data_analysis_agent/business_tools.py) · [策略与结论协议](src/data_analysis_agent/business_investigation.py) · [数据生成与题库](src/data_analysis_agent/business_cases.py) · [评分与预算内配对](src/data_analysis_agent/business_benchmark.py)

## Mode B 晚到率实验（历史）

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

该版将测试合并为 7 个文件、260 个用例，覆盖安全边界、关键回归及线上协议。

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
