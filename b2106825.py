from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = joblib.load('best_decision_tree_model_iris.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Tạo mảng numpy từ dữ liệu
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Dự đoán loài hoa
            prediction = model.predict(input_features)

            # Dịch kết quả dự đoán sang tên loài hoa
            iris_species = ['Setosa', 'Versicolor', 'Virginica']
            predicted_species = iris_species[prediction[0]]

            # Trả về kết quả trên trang result.html
            return render_template('result.html', predicted_species=predicted_species)

        except Exception as e:
            return f"Có lỗi xảy ra: {e}"

if __name__ == "__main__":
    app.run(debug=True)
