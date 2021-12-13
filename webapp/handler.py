import os
import pickle as pi
import pandas as pd
from flask import Flask, request, Response
from riscodecredito.Riscodecredito import riscodecredito

# Carregando Modelo
modelo = pi.load(open('modelo/random_forest_best.pkl', 'rb'))

# Inicializando API
app = Flask(__name__)
@app.route('/riscodecredito/predicao', methods=['POST'])

def risco_de_credito_predicao():
    teste_json = request.get_json()
    
    if teste_json: 
        if isinstance(teste_json, dict): # Existe um dado
            teste_raw = pd.DataFrame(teste_json, index=[0])
        
        else: # Multiplos dados
            teste_raw = pd.DataFrame(teste_json, columns=teste_json[0].keys())
            
        # Instanciar classe credito
        pipeline = riscodecredito()
        
        # descrição dos dados
        df1 = pipeline.descricao_dados(teste_raw)
        
        # preprocessamento_1
        df2 = pipeline.pre_processamento_1(df1)
        
        # preprocessamento_2
        df3 = pipeline.pre_processamento_2(df2)
        
        # Predições
        df_response = pipeline.get_prediction(modelo, teste_raw, df3)
        
        return df_response
         
    else:
        return Response('{}', status=200, mimetype='application/json')
    

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
