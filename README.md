### 1. **模型开发**

- 运行样例代码`penalized_linear.py`
  - 确认代码处理了数据标准化、滞后变量、样本分割（训练、验证、测试）。
  - 掌握 Lasso、Ridge、Elastic Net 模型的使用和超参数调优过程。
- 思考改进方向，例如：
  - 引入其他机器学习模型（随机森林、XGBoost、神经网络等）。
  - 使用特征选择方法（如 PCA 或基于特征重要性的选择）减少噪声。

------

### 2. **投资组合构建**

- `portfolio_analysis.py`：如何基于预测结果构建投资组合。
- 确保投资组合的设计符合任务要求（每月 50-100 支股票、定期再平衡）。
- 改进投资组合策略，例如：
  - 优化股票选择规则（如权重分配、风险控制）。
  - 添加多样化或风控机制（如最大持仓权重限制）。

------

### 3. **模型评估和回测**

- 使用 Out-of-Sample R² 评估预测模型性能。
- 评估投资组合表现，计算以下指标：
  - 夏普比率、累计收益率、平均收益率、波动率、最大回撤、单月最大损失。
- 绘制关键图表（如累计收益曲线、分位收益比较）。

### /penalized_linear
- penalized_linear.py: the original version provided by the teacher
- penalized_linear-v1.py: the first version, changed in mainly these modules
  - 新增模型
    - 添加 Random Forest 和 Gradient Boosting 回归模型
    - 使用 sklearn.ensemble 中的 RandomForestRegressor 和 GradientBoostingRegressor
  - 超参数调优
    - 为每个模型设计了超参数搜索流程，通过验证集优化超参数
  - 将新模型的预测结果存储到 output.csv，并计算 Out-of-Sample R^2
- penalized_linear-v2.py: the second version, changed in mainly features
  - 使用 RandomForestRegressor 获取特征重要性，选取最重要的 20 个特征
  - 对选定特征进行主成分分析（PCA），降维至 10 个主成分
  - 将所有模型的输入调整为 PCA 降维后的特征
- penalized_linear-v3.py: the third version, optimize
  - 引入 GridSearchCV
    - 使用网格搜索优化随机森林的超参数，包括 n_estimators 和 max_depth
    - 确保最佳参数通过交叉验证选择
  - 引入 BayesSearchCV
    - 使用贝叶斯优化对梯度提升模型进行超参数调优
    - 优化参数包括 n_estimators、learning_rate 和 max_depth
  - 性能提升
    - 网格搜索和贝叶斯优化结合更有效率地探索参数空间，确保模型在验证集上表现最佳
- penalized_linear-v4.py: the fourth version, data process
  - 数据平滑处理
    - 使用 Savitzky-Golay 滤波器对特征数据进行平滑，以减少噪声干扰
  - 交叉验证
    - 引入 KFold 交叉验证，用于随机森林和梯度提升模型的超参数优化
    - 提高模型的稳定性和泛化性能

### /portfolio_analysis

为了观察因子对股票价格的影响显著性以及IC的表现，可以按照以下步骤进行分析并结合已有的代码进行实现：

### 1. **数据准备**
确保在`penalized_linear.py`中提取到所有因子的预测值，生成`output.csv`文件。这个文件包含每只股票每个月的预测收益和真实收益。

- 因子名在`factor_char_list.csv`中，可以在`penalized_linear.py`中动态加载这些因子用于建模。
- 标准化因子值（例如`rank transform`至`[-1, 1]`）以确保数据一致性。

### 2. **计算信息系数（IC）**
IC是预测值与实际收益的横截面相关系数：
能够帮助判断因子的预测能力，IC绝对值越大说明因子越有效。

### 3. **因子筛选**
根据IC值筛选有效因子：
- 设置阈值（例如`mean_IC > 0.02`），筛选出表现良好的因子。
- 根据IC正负决定因子方向：正相关的因子值越大越好，负相关的因子值越小越好。

### 4. **优化投资组合**
利用`portfolio_analysis.py`中提供的组合构建框架：
- 将筛选后的因子应用于组合策略。
- 确保满足投资组合中股票数量限制（50到100之间），并在每个月重新平衡。

### 5. **性能评估**
结合Sharpe Ratio、最大回撤等指标对组合进行评估。

### 6. **改进思路**
- **非线性模型**：在`penalized_linear.py`中引入如XGBoost、Random Forest等非线性模型。
- **多因子组合**：对不同因子赋权，利用优化算法选择最优因子权重。