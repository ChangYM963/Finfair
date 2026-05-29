# FinFair 公开演示 Demo

English version: [README.md](README.md)

这是论文 **FinFair: A Regulatory-Aligned Robust Optimization Framework for Fair and Lightweight Financial Decision Models** 的公开演示包。它用于展示论文的核心问题、数学建模、方法结构、反事实数据格式、评估指标和主要实验发现。

这个 demo 的定位是 **method-oriented public package**：代码能体现 FinFair 的基本思路，但不会泄露完整工程实现。仓库中只包含样例数据、指标计算、方法 skeleton 和论文结果图；完整训练流程、完整 HMA-BDE prompts、全量数据、checkpoint、训练日志和私有工程细节均被有意省略。

## 1. 研究问题

金融决策支持系统需要满足一致性、可解释性和非歧视性要求。对于投资建议、适当性评估、资产配置等场景，模型输出不应因为客户的性别、年龄、地区等 **非经济属性** 改变而发生不合理变化。

论文关注一个现实约束：金融机构往往不能直接部署大型 LLM，因为它们存在计算成本高、审计困难、数据出境或隐私合规风险等问题。因此，更可落地的选择是 0.1B 到 0.3B 参数规模的轻量模型。但轻量模型容量有限，更容易吸收训练数据中的人口属性相关伪相关，从而在反事实场景下输出不稳定。

FinFair 的核心目标是：

> 在保持金融推理准确性的同时，让轻量模型对非经济人口属性扰动保持稳定。

## 2. Counterfactual Invariance 定义

论文将公平性表述为一种 **demographic perturbation 下的预测不变性**。设同一金融场景 $m$ 存在两个只改变人口属性描述的变体 $x_m^{(v_1)}$ 与 $x_m^{(v_2)}$，理想情况下模型应满足：

$$
f_\theta(x_m^{(v_1)}) = f_\theta(x_m^{(v_2)}),
\qquad
\forall (v_1,v_2)\in \mathcal{V}\times\mathcal{V},\ \forall m.
$$

其中 $\mathcal{V}$ 是人口属性扰动集合，例如 gender、age、region。若同一金融语义下，仅因属性描述不同导致模型预测变化，则说明模型可能利用了与金融基本面无关的人口属性线索。

论文进一步定义 regulatory-aligned feasible set：

$$
\mathcal{F}_{\mathrm{reg}}
=
\left\{
\theta :
f_\theta(x^{(v_1)}) = f_\theta(x^{(v_2)}),
\ \forall (v_1,v_2)\in\mathcal{V}\times\mathcal{V},\ \forall x
\right\}.
$$

考虑到训练过程中的随机性，论文使用 $\epsilon$-relaxed feasibility region：

$$
\mathcal{F}_{\mathrm{reg}}(\epsilon)
=
\left\{
\theta :
\mathbb{E}_{(x,v_1,v_2)}
\left[
\| f_\theta(x^{(v_1)}) - f_\theta(x^{(v_2)}) \|
\right]
\le \epsilon
\right\}.
$$

这个定义把“公平性”转化为一个 robust optimization 中的可行域约束。

## 3. Robust Optimization 视角

如果 $\mathcal{D}$ 表示金融场景分布，$\mathcal{V}$ 表示人口属性扰动集合，那么稳健金融决策规则可以写作：

$$
\min_{\theta}
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\left[
\max_{v\in\mathcal{V}}
\ell_{\mathrm{task}}\big(f_\theta(x^{(v)}), y\big)
\right].
$$

进一步加入 regulatory feasibility 后，论文中的理想目标为：

$$
\min_{\theta \in \mathcal{F}_{\mathrm{reg}}(\epsilon)}
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\left[
\max_{v\in\mathcal{V}}
\ell_{\mathrm{task}}(f_\theta(x^{(v)}), y)
\right].
$$

直接优化这个目标很难，所以 FinFair 用三个可微模块构造可训练近似：

- 主任务目标控制金融推理准确性；
- 对抗模块近似 demographic perturbation 下的 min-max invariance；
- teacher-guidance 模块作为 soft projection，把 student 拉向 rational decision set。

## 4. FinFair 方法结构

FinFair 由三个模块组成：

![FinFair framework](assets/framework.png)

### 4.1 Main-Task-Head

轻量 encoder 产生表示 $h(x;\theta)$，多选任务头输出：

$$
z^y = W_y h(x;\theta) + b_y,
\qquad
p_\theta(y\mid x)=\mathrm{softmax}(z^y).
$$

主任务 loss 是标准交叉熵：

$$
L_{\mathrm{main}}
=
-\mathbb{E}_{(x,y)\sim\mathcal{D}}
\log p_\theta(y\mid x).
$$

这一项保证模型仍然学习金融决策任务本身，而不是只追求形式上的一致性。

### 4.2 Adversarial-Bias-Head

对抗头尝试从表示 $h(x;\theta)$ 中预测敏感属性 $b$。通过 Gradient Reversal Layer，encoder 在反向传播中被迫去除可被属性分类器利用的人口属性信息。

对抗 loss 为：

$$
L_{\mathrm{adv}}
=
-\mathbb{E}_{(x,b)\sim\mathcal{D}}
\log p_\phi\big(b \mid \mathrm{GRL}(h(x;\theta))\big).
$$

对应的 saddle-point 形式是：

$$
\min_\theta \max_\phi
\mathbb{E}_{(x,b)\sim\mathcal{D}}
\left[
\ell_{\mathrm{adv}}\big(\phi(h(x;\theta)), b\big)
\right].
$$

直观上，$\phi$ 尽力识别敏感属性，$\theta$ 则学习让表示对敏感属性不可识别。

### 4.3 Teacher-Guidance-Module

单纯对抗去偏可能会过度删除有用金融信息，尤其是在轻量模型容量有限时。FinFair 因此引入 rational teacher。teacher 在 curated unbiased financial questions 上训练，然后被冻结，用其输出分布 $p_T(y\mid x)$ 约束 student：

$$
L_{\mathrm{distill}}
=
\mathrm{KL}
\left(
p_T(y\mid x)
\;\|\;
p_\theta(y\mid x)
\right)
=
\sum_y p_T(y\mid x)
\log
\frac{p_T(y\mid x)}{p_\theta(y\mid x)}.
$$

这一项相当于一个 soft constraint，使 student 的决策分布靠近 rational decision set，避免对抗训练牺牲金融合理性。

### 4.4 Unified Objective

论文中的多目标优化形式为：

$$
\min_\theta
\left[
L_{\mathrm{main}}
+
\alpha_{\mathrm{adv}} L_{\mathrm{adv}}^\star
+
\beta_T L_{\mathrm{distill}}
\right],
\qquad
L_{\mathrm{adv}}^\star = \max_\phi L_{\mathrm{adv}}(\theta,\phi).
$$

实际训练中使用 GRL 近似 saddle point，显式总损失为：

$$
L_{\mathrm{total}}
=
L_{\mathrm{main}}
+
\alpha_{\mathrm{adv}} L_{\mathrm{adv}}
+
\beta_T L_{\mathrm{distill}}.
$$

公开 skeleton 代码位于 [`src/finfair_skeleton.py`](src/finfair_skeleton.py)。它只保留上述目标函数和模块接口，不包含真实模型选择、tokenizer、collator、优化器、训练循环、GPU 配置或 checkpoint 逻辑。

## 5. HMA-BDE 数据构造

论文还提出 HMA-BDE，用于构造 attribute-controlled counterfactual financial pairs。基本流程是：

1. 从金融多选决策问题中选择容易产生偏见的 seed scenarios；
2. 通过人工修改形成高质量 seed pairs；
3. 使用 LLM 进行受控扩展；
4. 通过自动过滤与人工复筛减少语义漂移和类别混杂；
5. 形成按 `base_id` 分组的反事实评估集。

![HMA-BDE data process](assets/data_process.png)

demo 中的 JSONL 样例格式如下：

```json
{
  "id": "SCN001_v1",
  "base_id": "SCN001",
  "variant": "gender_female",
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B"
}
```

同一 `base_id` 下的样本是同一金融场景的不同 demographic variants。demo 只提供少量人工整理样例，不包含完整 HMA-BDE 数据、完整 prompt 或过滤规则。

## 6. 评估指标

论文不只看单样本准确率，而是同时评估“是否答对”和“同一题组是否稳定”。

### Sample-level Accuracy

$$
\mathrm{Acc}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\hat{y}_i = y_i
\right].
$$

### Intra-group Consistency

对每个反事实题组 $G_m$，若组内所有 variants 的预测一致，则该题组 consistent：

$$
\mathrm{Cons}
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}
\left[
|\{ \hat{y}_i : i\in G_m \}| = 1
\right].
$$

### Consistency-Correctness

题组不仅要预测一致，还要预测正确：

$$
\mathrm{CC}
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}
\left[
|\{ \hat{y}_i : i\in G_m \}| = 1
\ \land\
\hat{y}_{G_m}=y_{G_m}
\right].
$$

指标实现见 [`src/metrics.py`](src/metrics.py)。

## 7. 运行 Demo

在 `submission_demo/` 目录下运行：

```bash
python run_demo.py
```

输出示例：

```text
Metric summary
-------------------------------------------------------------------------------------------
method       | sample_accuracy | intra_group_consistency | consistency_correctness | groups
-------------------------------------------------------------------------------------------
baseline     | 0.500           | 0.000                   | 0.000                   | 3
finfair_demo | 1.000           | 1.000                   | 1.000                   | 3
```

这里的 `finfair_demo` 是准备好的示例预测，用于展示论文中的评估协议。它不是公开 checkpoint 的实时推理结果。

## 8. 论文实验结果概览

### Baseline vs. FinFair

![Baseline vs FinFair](assets/baseline_vs_finfair.png)

论文中，普通轻量监督模型在反事实一致性上表现较弱，而 FinFair 显著提升 consistency 和 consistency-correctness。

### 轻量模型与 LLM 对比

![Seven model comparison](assets/7_model_vs.png)

论文报告的主要对比结果包括：

| Model | Params | Consistency | Sample Acc. | Consistent-Correct |
|---|---:|---:|---:|---:|
| Electra FinFair | 0.012B | 0.794 | 0.762 | 0.667 |
| DeBERTa FinFair | 0.086B | 0.762 | 0.794 | 0.683 |
| MacBERT FinFair | 0.102B | 0.754 | 0.786 | 0.675 |
| BERT FinFair | 0.110B | 0.905 | 0.849 | 0.810 |
| RoBERTa FinFair | 0.325B | 0.921 | 0.857 | 0.825 |

论文的核心结论是：在同一反事实评估协议下，轻量 FinFair 模型可以在稳定性指标上达到或超过更大的通用 LLM。

### 消融实验

![FinFair ablation](assets/finfair_ablation.png)

| Variant | Consistency | Sample Acc. | Consistent-Correct |
|---|---:|---:|---:|
| BERT baseline | 0.190 | 0.389 | 0.063 |
| Adv-only | 0.270 | 0.437 | 0.127 |
| Teacher-only | 0.794 | 0.873 | 0.778 |
| Full FinFair | 0.905 | 0.849 | 0.810 |

结果说明：teacher-guided rational alignment 提供主要稳定性提升，对抗模块进一步抑制 demographic leakage；二者组合效果最好。

### 蒸馏策略比较

![FinFair distillation](assets/finfair_distillation.png)

| Distillation Strategy | Consistency | Sample Acc. | Consistent-Correct |
|---|---:|---:|---:|
| Label-smoothing | 0.635 | 0.730 | 0.571 |
| Hard-label | 0.714 | 0.746 | 0.619 |
| Soft + Entropy | 0.877 | 0.845 | 0.790 |
| KL soft | 0.905 | 0.849 | 0.810 |

KL soft distillation 保留 teacher 输出分布的完整结构，包括不确定性、选项间关系和偏好排序，因此最能传递 rationality signal。

## 9. 公开代码边界

这个 demo 公开：

- 样例反事实数据格式；
- 指标计算逻辑；
- FinFair 三组件方法 skeleton；
- 论文结果图与简要分析。

这个 demo 不公开：

- 完整 HMA-BDE prompts、过滤规则和扩展脚本；
- 完整训练集、验证集、测试集；
- 真实模型 checkpoint、训练日志和硬件配置；
- tokenizer、collator、optimizer、分布式训练、多 GPU 等工程实现；
- 超参数搜索和私有实验脚本。

这样设计的目的，是在 GitHub 上清楚展示论文内容，同时保护完整工程细节。
