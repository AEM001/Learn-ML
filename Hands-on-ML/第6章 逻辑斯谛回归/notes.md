# 逻辑斯谛函数
def logistic(z):
    # $\sigma(z) = \frac{1}{1 + e^{-z}}$
    return 1 / (1 + np.exp(-z))

def GD(num_steps, learning_rate, l2_coef):
    # 初始化模型参数
    theta = np.random.normal(size=(X.shape[1],))
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    for i in range(num_steps):
        pred = logistic(X @ theta)  
        # $\hat{y} = \sigma(X\theta)$
        grad = -X.T @ (y_train - pred) + l2_coef * theta  
        # $\nabla J(\theta) = -X^T(y - \hat{y}) + \lambda\theta$
        theta -= learning_rate * grad  
        # $\theta \leftarrow \theta - \eta\nabla J(\theta)$
        # 记录损失函数
        train_loss = - y_train.T @ np.log(pred) \
                     - (1 - y_train).T @ np.log(1 - pred) \
                     + l2_coef * np.linalg.norm(theta) ** 2 / 2  
                     # $J(\theta) = -[y^T\log(\hat{y}) + (1-y)^T\log(1-\hat{y})] + \frac{\lambda}{2}||\theta||^2$
        train_losses.append(train_loss / len(X))
        test_pred = logistic(X_test @ theta)
        test_loss = - y_test.T @ np.log(test_pred) \
                    - (1 - y_test).T @ np.log(1 - test_pred) 
                     # $J_{test}(\theta) = -[y_{test}^T\log(\hat{y}_{test}) + (1-y_{test})^T\log(1-\hat{y}_{test})]$
        test_losses.append(test_loss / len(X_test))
        # 记录各个评价指标，阈值采用0.5
        train_acc.append(acc(y_train, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))
        train_auc.append(auc(y_train, pred))
        test_auc.append(auc(y_test, test_pred))
        
    return theta, train_losses, test_losses, \
    train_acc, test_acc, train_auc, test_auc