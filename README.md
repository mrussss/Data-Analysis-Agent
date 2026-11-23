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

离线是默认路径。可用 `OPENAI_API_KEY`、`OPENAI_MODEL`、可选 `OPENAI_BASE_URL` 加 `--online` 复跑。每次 CLI 运行共享请求与 token 预算，默认上限分别为 60 与 250,000，SDK 自动重试关闭。预算内仅对临时网络故障、超时、429 和 5xx 按退避重试至多两次。每次尝试均占请求名额，可能已送达而用量未知时保留预留额度。成功响应缺少用量时停止。

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

### 真实模型对照

DeepSeek 请求统一启用思考并设置 `reasoning_effort=low`。业务决策和报告最多 3,072 个输出 token。动作提示列明当前可用工具和 Evidence ID，模型必须复制合法工具 ID。每个调查最多额外纠正两次 JSON 格式、恢复一次明确截断。动作语义仍按原校验，最多另给一次动作纠正。Trace 保留脱敏动作、错误类别、各次恢复调用和用量。模型可在停止时一并提交报告，所引用的证据仍由原评分器核查。两组共用动作预算和评分。如果两组用到不同模型版本，或同一组混用版本，保留原始记录并排除该题的公平配对成绩。 [DeepSeek 官方思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)

更早的历史试跑有未核对用量，仍保留在各自的 Trace。首次协议评测单独记账，后续 D 类和可靠性复测接续该账本。新 Low 探针在 VX Flash 返回合法 JSON，消耗 88 token，观察到推理内容。网关没有回报实际推理等级。 [历史开发集 Trace](evaluation/business_online_dev_wsl.json) · [本轮开发集 Trace](evaluation/business_online_dev_protocol.json)

开发集四题均完成两组真实模型运行，所有响应均通过动作和结论格式校验。Frozen 答对 3/4，Adaptive 答对 2/4。Adaptive 的促销题核实了实付价格和促销占比，却少查了公开事件。两组在事后事件题都查到事件晚于销量下跌并拒绝归因，但都漏引销量下降这条必答证据。开发集含探针共 19 次请求、47,875 个已计量 token。由此验证规划和执行协议可以进入冻结配对，独立评分保留上述失分。

冻结集依序完成 A、B、C 三道同模型配对。各模型响应均由 VX Flash 提供，未观察到模型版本混用。C 题的 Adaptive 响应以 `length` 结束，计为该组失败。该次请求的输出上限为 3,072 token，VX 回报 6,042 个 completion token。进入 D 类题时剩余 5 次请求，低于预先设定的 7 次完整配对启动额度，因此两道 D 题保留为未运行。 [冻结集逐案 Trace](evaluation/business_online_test_protocol.json)

| 策略 | 答对 | 完成报告 | 调查动作 | 已计量 Token | 总耗时 | 错误归因 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Workflow | 3/3 | 3/3 | 11 | 0 | 0.42 秒 | 0 |
| Rule-Adaptive | 3/3 | 3/3 | 11 | 0 | 0.33 秒 | 0 |
| LLM-Frozen | 2/3 | 3/3 | 11 | 19,455 | 144.01 秒 | 0 |
| LLM-Adaptive | 1/3 | 2/3 | 8 | 32,937 | 146.14 秒 | 0 |

本轮合计 35 次请求、100,267 个已计量 token，全部有用量记录。Adaptive 的 B 题只查了晚到组与准时组的评价差异，漏掉两期晚到率和差评率各自的变化。C 题第一步按 SKU 定位销量缺口，接着核实目标 SKU 在两期均有货，最后查到针对同一 SKU、日期吻合的召回通知。这个路径确实使用了新证据，但库存证据反驳缺货假设前，模型没有留下可比较的后续计划，严格反证重规划为零。Rule-Adaptive 已能拿齐这些证据及投诉率，当前样本没有观察到模型的净收益。

离线将原题原样压到三次动作后，Fixed Workflow 和 Rule-Adaptive 均从 30/30 降到 24/30，其中 A 8/8、B 7/8、C 3/8、D 6/6。五次预算下每组只有五题用满。14 道引用本地档案的题中，12 题各有一条记录，1 题有两条，1 题为空。题面经常点明应查指标或来源，所需证据通常可以提前确定。

### D 类续跑与独立召回复测

最初被截断的召回请求已经使用 3,072 的上限。其返回正文为空，结束原因为 `length`，VX 回报 6,042 个 completion token。这个记录无法区分推理占用和网关计量方式。续跑维持上限，缩短协议与报告输入，并独立计量每次恢复。

两道预选 D 类题均完成同模型首次配对，两组规则均答对，两组模型各答对 1/2。在缺货题中，Adaptive 从单位投诉率稳定转查库存，发现销量降幅集中在已缺货 SKU，再核实同期报道。它以 4 次动作答对，Frozen 用了 5 次但漏查报道。两组规则同样用 4 次动作答对。时间错配题的 Adaptive 查齐证据，正文也拒绝了事后事件解释，但输出 `association`，原评分要求 `insufficient`。 [D 类首次配对](evaluation/business_online_test_d.json)

| 策略 | 首次冻结答对 | 首次完成报告 | 首次调查动作 | 首次 Token | 首次总耗时 | 召回复测答对 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Workflow | 5/5 | 5/5 | 19 | 0 | 0.49 秒 | 1/1 |
| Rule-Adaptive | 5/5 | 5/5 | 19 | 0 | 0.41 秒 | 1/1 |
| LLM-Frozen | 3/5 | 5/5 | 21 | 34,301 | 258.34 秒 | 0/1 |
| LLM-Adaptive | 2/5 | 4/5 | 15 | 62,420 | 333.06 秒 | 0/1 |

首次冻结合计覆盖 A、B、C 和两道 D，A–C 沿用恢复机制调整前的首次结果。召回复测单独记录。两组模型都完成报告，但均未用工具核实销量实际集中下降的 SKU。D 类与复测新增 22 次请求、76,388 个已计量 token。22 次中没有 JSON 解析错误，发生 4 次明确截断，均在一次恢复后取得合法输出。4 次恢复请求消耗 6,457 token，4 个被截断的原始请求消耗 28,984 token。所有配对仍使用 VX Flash。续跑后的累计为 57 次请求、176,655 个已计量 token。 [召回复测](evaluation/business_online_reliability_retest.json)

当前续跑沿用此前两份账本的 35 次请求和 100,267 个已计量 token。整轮上限为 80 次实际请求和 260,000 个已计量 token，每题 5 次工具动作。环境变量 `MAX_LLM_REQUESTS`、`MAX_LLM_TOTAL_TOKENS` 可以调低整轮上限。每次出站请求先写入 `data/processed/.business_online_usage.json`，响应返回后按实际用量结算。未返回用量则保留预留。该文件是同一工作区跨命令的预算记录，续跑时保留它。重复传入历史结果会去重。旧轮未知用量留在旧记录中。

首次冻结续跑默认跳过已经尝试的题。复跑召回题需要显式标为可靠性复测并使用新输出路径。两类结果均保留逐案 Trace。下方命令对应 D 类首次配对和独立标记的召回复测。再次续跑时应选择新的输出文件。

```bash
python -m data_analysis_agent eval --business --online \
  --prior-results evaluation/business_online_dev_protocol.json \
  --prior-results evaluation/business_online_test_protocol.json \
  --output evaluation/business_online_test_d.json
python -m data_analysis_agent eval --business --online --reliability-retest --case recall_scope \
  --prior-results evaluation/business_online_dev_protocol.json \
  --prior-results evaluation/business_online_test_protocol.json \
  --prior-results evaluation/business_online_test_d.json \
  --output evaluation/business_online_reliability_retest.json
```

[工具与 SQL](src/data_analysis_agent/business_tools.py) · [策略与结论协议](src/data_analysis_agent/business_investigation.py) · [数据生成与题库](src/data_analysis_agent/business_cases.py) · [评分与在线配对](src/data_analysis_agent/business_benchmark.py)

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
