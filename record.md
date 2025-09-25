# 2025.09.21
### xgb:
score: 0.6558
param: {'classifier__learning_rate': 0.05, 'classifier__max_depth': 4, 'classifier__n_estimators': 500, 'classifier__scale_pos_weight': 5}

### rf:
✅ 最优参数: {'classifier__max_depth': 10, 'classifier__n_estimators': 500}
✅ 最优分数: 0.6405

### lgb:
✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 4, 'classifier__n_estimators': 500}
✅ 最优分数: 0.6561


# 2025.09.22
### lgb:
✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 4, 'classifier__n_estimators': 500}
✅ 最优分数: 0.6739

# 2025.09.23

### lgb
✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 4, 'classifier__n_estimators': 500}
✅ 最优分数: 0.6775

✅ 最优参数: {'classifier__class_weight': 'balanced', 'classifier__learning_rate': 0.06, 'classifier__max_depth': 3, 'classifier__n_estimators': 600}
✅ 最优分数: 0.6758

#### 增加矩阵分解
 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__n_estimators': 700}
✅ 最优分数: 0.6778

✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__n_estimators': 1100}
✅ 最优分数: 0.6781


#### 改良矩阵分解
✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__n_estimators': 1100}
✅ 最优分数: 0.6829


### 增加先验概率特征
✅ 最优参数: {'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__n_estimators': 1100}
✅ 最优分数: 0.6837