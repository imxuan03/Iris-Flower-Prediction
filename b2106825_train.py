from sklearn import datasets
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Tải dữ liệu Iris
iris = datasets.load_iris()
columns = ["Petal length", "Petal Width", "Sepal Length", "Sepal Width"]
df = pd.DataFrame(iris.data, columns=columns)
y = iris.target

best_accuracy = 0
best_model = None

# Thực hiện huấn luyện mô hình 10 lần
for i in range(10):
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo tỷ lệ 70/30
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
    
    # Xây dựng mô hình Decision Tree
    model = DecisionTreeClassifier()
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Tính độ chính xác
    accuracy = model.score(X_test, y_test)
    print(f"Lần {i+1}: Độ chính xác của mô hình Decision Tree: {accuracy:.3f}")
    
    # Kiểm tra nếu mô hình hiện tại có độ chính xác lớn hơn
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Sau khi thực hiện xong 10 lần, lưu mô hình có độ chính xác lớn nhất
if best_model is not None:
    model_filename = 'best_decision_tree_model_iris.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Mô hình có độ chính xác cao nhất ({best_accuracy:.5f}) đã được lưu thành file: {model_filename}")
