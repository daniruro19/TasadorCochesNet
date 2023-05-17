from flask import Flask, request
import pickle
from flask_cors import CORS
import json
import os
import tensorflow as tf
import pandas as pd

application = Flask(__name__)
CORS(application)

@application.route('/flask', methods=['GET'])
def flask():
    return '<p><a href="https://flask.palletsprojects.com/en/2.2.x/">https://flask.palletsprojects.com/en/2.2.x/</a></p>'

@application.get('/api/get')
def get_method():
    word = request.args.get('word', '<no word>')
    return {
        'hello': 'hello, ' + word
    }

@application.route('/')
def main():
    return '<p>Hello, World!</p>'

@application.get('/car')
def car():
    neural_model = tf.keras.models.load_model('cochesNet\\app\\modelo_keras_coches_net.hdf5')
    
    with open('cochesNet\\app\\dict_vectorizer_coches_net.pck', 'rb') as file:
     dv = pickle.load(file)

    campos_numericos = ['km', 'year', 'cubicCapacity', 'doors', 'hp']

    # obtenemos el coche de la request

    coche = json.loads(request.args.get('coche', ''))
    coche['year'] = float(coche['year'])
    coche['hp'] = float(coche['hp'])
    coche['km'] = float(coche['km'])
    coche['doors'] = float(coche['doors'])
    coche['cubicCapacity'] = float(coche['cubicCapacity'])
    coche_df = pd.DataFrame(coche, index=[0])
    coche_dv = dv.transform(coche_df.to_dict(orient='records'))
    try:
        precio = neural_model.predict(coche_dv).tolist()
    except:
        print('malo')
        precio = [0]
    
    print(precio)

    return {
       'precio': precio[0]
    }


application.run()