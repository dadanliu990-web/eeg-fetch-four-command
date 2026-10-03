# 单受试者 EEG 四指令控制 Fetch：离线仿真原型

本项目使用 **BCI Competition IV 2a** 的提示式运动想象 EEG，识别左手、右手、双脚和舌头四类任务，并把分类结果映射为 FetchPickAndPlace 仿真中的左移、右移、抓取和放置指令。当前使用公开数据中同一位受试者 **A07** 的离线试次；机器人抓放轨迹由规则控制。

> 项目范围：离线 EEG 分类与仿真回放。它没有实时 EEG 输入、持续监听时的静息／无指令检测，也未连接实物机械臂。

## 指令映射

| 运动想象类别 | 仿真指令 | 执行方式 |
| --- | --- | --- |
| 左手 `LEFT_HAND` | `LEFT_SHIFT` | 夹爪沿仿真 `+y` 移动约 4 cm |
| 右手 `RIGHT_HAND` | `RIGHT_SHIFT` | 夹爪沿仿真 `-y` 移动约 4 cm |
| 双脚 `FEET` | `GRASP` | 接近方块、闭合夹爪、确认方块升高 |
| 舌头 `TONGUE` | `RELEASE` | 将方块移到目标、放下并松开夹爪 |

每个有提示的 EEG 试次最多产生一次指令。夹爪已持物时忽略重复抓取；未持物时忽略释放。最高类别概率低于验证集选定的阈值时执行 `NO_ACTION`。这些状态约束不会纠正 EEG 预测本身。

## 数据与方法

数据来自 [BCI Competition IV 2a 官方下载页](https://bbci.de/competition/iv/download/)；[数据说明](https://www.bbci.de/competition/iv/desc_2a.pdf)介绍了四类提示任务、22 个 EEG 通道、3 个 EOG 通道及伪迹标记。评估会话的真实标签见[竞赛结果页](https://bbci.de/competition/iv/results/)。请自行从官方渠道获取数据并遵守其使用条件；仓库不包含原始 EEG 文件或标签文件。

1. 只取前 22 个 EEG 通道，不使用 EOG 通道。剔除官方标记的伪迹试次。
2. 从提示后 0.5 秒开始提取 2 秒窗口，使用五个 8–30 Hz 滤波频带。各实验轮次分别进行前向因果滤波。
3. 每个频带提取 CSP 特征，拼接后用 LDA 进行四分类。
4. 对 A01–A09：使用各自训练会话的第 1–4 轮拟合、第 5–6 轮验证；仅依据验证集平衡准确率选择受试者。此次实验选中 A07。
5. 用 A07 完整训练会话重新拟合分类器，最后才在 A07E 评估会话上评估。
6. 单独使用 A07T 第 1–4 轮拟合临时模型、第 5–6 轮选择“不动作”阈值；A07E 不用于选阈值。阈值从预先列出的候选值中选取：要求验证集保留指令正确率至少 90%、覆盖率至少 50%，取满足条件的最低阈值。

## 本次实验结果

| 指标 | 结果 |
| --- | ---: |
| 选中受试者 | A07 |
| A07E 有效测试试次 | 277 |
| 四类平衡准确率 | 0.805 |
| 未加不动作规则的正确预测 | 222/277 |
| 验证集选定的不动作阈值 | 0.80 |
| 加阈值后发出指令 | 224/277 |
| 已发出指令中的正确预测 | 187/224，83.5% |
| 被阈值拦下的错误／正确预测 | 18／35 |
| 仍被执行的错误预测 | 37 |

加阈值后的完整 Fetch 回放中，执行了 30 次抓取和 29 次成功放置。放置次数受随机排列的 EEG 试次及状态约束影响，**不能解释为脑控成功率**。A07T 验证集上保留指令正确率为 63/68（92.6%），但 A07E 上降为 187/224（83.5%）；分类器输出的概率并非可靠的安全保证。受试者选择和阈值选择都使用了 A07T 的第 5–6 轮，因此验证集结果还有选择偏倚；A07E 是未用于这两项选择的测试会话。

## Windows PowerShell 复现

建议 Python 3.12。EEG 训练与 Fetch 仿真可安装在两个独立虚拟环境中。以下命令在仓库根目录执行。

```powershell
py -3.12 -m venv .eeg_venv
& '.\.eeg_venv\Scripts\python.exe' -m pip install -r requirements-eeg.txt

py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-fetch.txt
```

把 `A01T.gdf`、`A01T.mat`、`A01E.gdf`、`A01E.mat`，依此类推直到 A09 的训练与评估文件，放在 `data\BCICIV_2a\`；也可以用 `--data-root` 指向其他目录。`.mat` 文件需包含 `classlabel` 真实标签。随后依次运行：

```powershell
# 选择受试者、训练最终模型、生成独立测试会话的逐试次预测
& '.\.eeg_venv\Scripts\python.exe' '.\select_bciciv2a_four_commands.py' --data-root '.\data\BCICIV_2a'

# 只用 A07T 训练轮次和验证轮次选低置信度不动作阈值
& '.\.eeg_venv\Scripts\python.exe' '.\calibrate_bciciv2a_A07_abstain.py' --data-root '.\data\BCICIV_2a'

# 默认读取目录中最新预测和同一受试者的最新阈值策略，回放前 12 条
& '.\.venv\Scripts\python.exe' '.\replay_bciciv2a_four_commands_fetch.py' --limit 12 --render

# 无图形窗口回放全部试次；--no-abstain 可与原方案对照
& '.\.venv\Scripts\python.exe' '.\replay_bciciv2a_four_commands_fetch.py' --limit 0
```

若本机已有多个预测结果，使用 `--csv` 指定相应 CSV，使用 `--policy` 指定 JSON 策略，避免误读其他实验文件。`--render` 会打开 MuJoCo 窗口，运行期间不要频繁点击窗口导致 Windows 报“未响应”；无窗口模式适合批量回放。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `select_bciciv2a_four_commands.py` | 受试者选择、FBCSP + LDA 训练、A07E 独立评估及预测导出 |
| `calibrate_bciciv2a_A07_abstain.py` | 仅用 A07T 验证轮次选择不动作阈值 |
| `replay_bciciv2a_four_commands_fetch.py` | 将离线预测按状态映射到 Fetch 仿真并生成回放日志 |
| `requirements-eeg.txt`、`requirements-fetch.txt` | 本次 Windows 环境所用依赖版本 |

模型、预测 CSV、策略 JSON、原始数据和虚拟环境均由程序在本机生成或由用户自行下载，并已加入 `.gitignore`。仓库不提供已训练模型；请按上述步骤复现。

## 局限

- 四类数据均为**有提示的运动想象**，没有可直接用于持续控制的“无指令”类别。
- 分类器在 A07E 上仍有高置信度错误；本项目不应用于无人工监督的实物机械臂操作。
- 抓取和放置是手工设计的宏动作，不是强化学习策略；仿真成功不等于实物成功。
- 这是单受试者离线实验，不代表其他受试者或实时 EEG 的表现。

## 参考资料

- [BCI Competition IV 2a 数据说明](https://www.bbci.de/competition/iv/desc_2a.pdf)
- [BCI Competition IV 数据下载](https://bbci.de/competition/iv/download/)
- [BCI Competition IV 评估标签与结果](https://bbci.de/competition/iv/results/)
- [Gymnasium Robotics 官方文档](https://robotics.farama.org/)
