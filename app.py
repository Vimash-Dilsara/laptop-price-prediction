from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the KNN regressor model
model = joblib.load('knn_regressor_model.pkl')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    ram = float(request.form['ram'])
    weight = float(request.form['weight'])
    company = request.form['company']
    cpu = request.form['cpu']
    gpu = request.form['gpu']
    opsyss = request.form['opsyss']
    typename = request.form['typename']
    
    # Handle checkbox values
    touchscreen = 1 if 'touchscreen' in request.form else 0
    ips = 1 if 'ips' in request.form else 0
    full_hd = 1 if 'full_hd' in request.form else 0

    # Convert categorical variables to one-hot encoding
    company_encoding = {
        'acer': [1, 0, 0, 0, 0, 0, 0, 0],
        'apple': [0, 1, 0, 0, 0, 0, 0, 0],
        'asus': [0, 0, 1, 0, 0, 0, 0, 0],
        'dell': [0, 0, 0, 1, 0, 0, 0, 0],
        'hp': [0, 0, 0, 0, 1, 0, 0, 0],
        'lenovo': [0, 0, 0, 0, 0, 1, 0, 0],
        'msi': [0, 0, 0, 0, 0, 0, 1, 0],
        'toshiba': [0, 0, 0, 0, 0, 0, 0, 1],
        'others': [0, 0, 0, 0, 0, 0, 0, 0]
    }

    cpu_encoding = {
        'amd': [1, 0, 0, 0],
        'intel_core_i3': [0, 1, 0, 0],
        'intel_core_i5': [0, 0, 1, 0],
        'intel_core_i7': [0, 0, 0, 1],
        'other': [0, 0, 0, 0]
    }

    gpu_encoding = {
        'amd_graphics': [1, 0, 0, 0],
        'arm_graphics': [0, 1, 0, 0],
        'intel_graphics': [0, 0, 1, 0],
        'nvidia_graphics': [0, 0, 0, 1]
    }

    opsyss_encoding = {
        'mac': [1, 0, 0],
        'windows': [0, 1, 0],
        'others': [0, 0, 1]
    }

    typename_encoding = {
        '2_in_1_convertible': [1, 0, 0, 0, 0, 0],
        'gaming': [0, 1, 0, 0, 0, 0],
        'netbook': [0, 0, 1, 0, 0, 0],
        'notebook': [0, 0, 0, 1, 0, 0],
        'ultrabook': [0, 0, 0, 0, 1, 0],
        'workstation': [0, 0, 0, 0, 0, 1]
    }

    company_encoded = company_encoding.get(company.lower(), [0, 0, 0, 0, 0, 0, 0, 0])
    cpu_encoded = cpu_encoding.get(cpu.lower(), [0, 0, 0, 0])
    gpu_encoded = gpu_encoding.get(gpu.lower(), [0, 0, 0, 0])
    opsyss_encoded = opsyss_encoding.get(opsyss.lower(), [0, 0, 0])
    typename_encoded = typename_encoding.get(typename.lower(), [0, 0, 0, 0, 0, 0])

# Create feature vector
    x_new = np.array([[ram, weight] + company_encoded + cpu_encoded + gpu_encoded + opsyss_encoded + typename_encoded + [touchscreen, ips, full_hd] + [0, 0]])

# Predict price
    predicted_price = model.predict(x_new)


    # Render the result
    return render_template('index.html', prediction_text=f'Predicted Price: ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
