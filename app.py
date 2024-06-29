# app.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
from network import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
digits = load_digits()
def one_hot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_reshaped = y.reshape(-1, 1)
    return encoder.fit_transform(y_reshaped)
X = digits.data
y = digits.target
y = one_hot_encode(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_network():
    data = request.json
    layers_info = data['layers_info']
    config = data['config']
    optimizer = config['optimizer']
    nn = MultiLayerNetwork(info=layers_info, optimizer=optimizer)
    result = nn.fit(X_train, y_train, epoch=config['epochs'], batch_size=config['batch_size'], learning_rate=config['learning_rate'], moment_const=config['momentum'])
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
