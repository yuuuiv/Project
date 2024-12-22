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


## 针对数据处理部分的优化

---

### **数据处理优化方向**
#### 1. **减少内存占用**
- **数据类型优化**：
  - 使用 `pandas` 的 `astype` 方法将数据类型压缩为最小必要类型。例如，将浮点数降级为 `float32`，将整型降级为 `int32`。
  ```python
  for col in df.select_dtypes(include=['float']).columns:
      df[col] = df[col].astype('float32')
  for col in df.select_dtypes(include=['int']).columns:
      df[col] = df[col].astype('int32')
  ```

- **按需加载**：
  - 使用 `chunksize` 按块加载数据，仅处理必要的字段或行。
  ```python
  chunks = pd.read_csv("stock_sample.csv", chunksize=100000)
  for chunk in chunks:
      process_chunk(chunk)  # 自定义数据处理函数
  ```

---

#### 2. **分布式处理**
- **并行化**：
  - 使用 `joblib` 或 `concurrent.futures` 并行处理每月数据。例如，按 `date` 分组后，将每组数据并行处理。
  ```python
  from joblib import Parallel, delayed

  def process_group(group):
      # 数据处理逻辑
      return processed_group

  results = Parallel(n_jobs=-1)(delayed(process_group)(group) for _, group in df.groupby("date"))
  processed_data = pd.concat(results)
  ```

- **分布式框架**：
  - 使用 `Dask` 或 `PySpark` 处理超大数据集。
  ```python
  import dask.dataframe as dd
  ddf = dd.read_csv("stock_sample.csv")
  ddf = ddf.groupby("date").apply(process_group, meta={...})
  ddf.compute()
  ```

---

#### 3. **特征工程优化**
- **矢量化计算**：
  - 避免 `for` 循环，尽量使用 `numpy` 或 `pandas` 的矢量化操作。例如：
  ```python
  # 现有代码逐列 rank，改为矢量化
  ranks = df[stock_vars].rank(axis=1, method='dense') - 1
  max_vals = ranks.max(axis=1)
  df[stock_vars] = (ranks.T / max_vals).T * 2 - 1
  ```

- **缺失值处理**：
  - 针对缺失值填充，先计算所有列的中位数，然后用矩阵操作一次性填充。
  ```python
  medians = df[stock_vars].median(skipna=True)
  df[stock_vars] = df[stock_vars].fillna(medians)
  ```

---

#### 4. **按需裁剪数据**
- **时间窗口过滤**：
  - 按时间分段仅加载需要的年份范围。
  ```python
  filtered_data = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2023-12-31")]
  ```

- **特征选择**：
  - 使用特征重要性筛选最相关的变量。例如，通过相关性过滤冗余特征：
  ```python
  corr_matrix = df[stock_vars].corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
  df = df.drop(columns=to_drop)
  ```

---

### **优化后的处理流程建议**
1. **预加载设置**：
   - 检查数据类型，优化内存占用；
   - 使用 `chunksize` 分块加载，减少内存压力。
2. **矢量化预处理**：
   - 使用矢量化处理变量的归一化和缺失值填充；
   - 统一对时间序列分段处理。
3. **分布式加速**：
   - 并行化分组处理，利用多核计算提升速度；
   - 数据规模过大时，使用 `Dask` 处理。
4. **输出存储优化**：
   - 保存中间处理结果为高效格式（如 `Parquet`）以减少后续读取时间。

---