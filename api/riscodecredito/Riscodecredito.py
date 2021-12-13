import pandas as pd
import numpy  as np
import inflection
import pickle as pi


class riscodecredito(object):
    
    def __init__(self):
        self.home_path = '/home/nickolas/Área de Trabalho/Portfólio/pa005/'
        self.age_scaler = pi.load(open(self.home_path + 'parametros/age_scaler.pkl', 'rb'))
        self.credit_amount_scaler = pi.load(open(self.home_path + 'parametros/credit_amount_scaler.pkl', 'rb'))
        
        
        
    def descricao_dados(self, df1):
        
        ## Renomeando Colunas
        colunas_velhas = ['Age', 'Sex', 'Job', 'Housing',
                          'Saving accounts', 'Checking account', 'Credit amount', 'Duration',
                          'Purpose']

        snakecase = lambda x: inflection.underscore(x)

        colunas_novas = list(map(snakecase, colunas_velhas))

        df1.columns = colunas_novas

        # Excluindo colunas 
        #df1 = df1.drop(['unnamed: 0', 'unnamed: 0.1'], axis=1)
    
        # Mudança Dados
        df1['job'] = df1['job'].astype('str')
    
        return df1
    
    def pre_processamento_1(self, df2):

        # Convertendo variável risk em binária
        #df2['risk'] = df2['risk'].apply(lambda x: 1 if x=='good' else 0)
        
        # Tratamento Dados Faltantes
        intervalo = (18, 25, 35, 60, 120)
        cats = ['Student', 'Young', 'Adult', 'Senior']
        df2['age_cat'] = pd.cut(df2.age, intervalo, labels=cats)
        
        df2['saving accounts'] = df2.groupby(['sex', 'age_cat'])['saving accounts'].transform(
        lambda x: x.fillna(x.mode()[0]))
        
        df2['checking account'] = df2.groupby(['sex', 'age_cat'])['checking account'].transform(
        lambda x: x.fillna(x.mode()[0]))
        
        df2['year'] = str(df2["duration"])
        df2.loc[df2["duration"] <= 24, "year"] = "0-2"
        df2.loc[(df2["duration"] > 24) & (df2["duration"] <= 48), "year"] = "2-4"
        df2.loc[(df2["duration"] > 48) & (df2["duration"] <= 72), "year"] = "4-6"
        #df3.loc[(df3["duration"] > 36) & (df3["duration"] <= 48), "year"] = "3-4"
        #df3.loc[(df3["duration"] > 48) & (df3["duration"] <= 60), "year"] = "4-5"
        #df3.loc[(df3["duration"] > 60) & (df3["duration"] <= 72), "year"] = "5-6"
        #df3.loc[(df3["duration"] > 72) & (df3["duration"] <= 84), "year"] = "6-7"
        
        df2['status'] = pd.qcut(df2['credit amount'],4, labels=['poor', 'mid', 'upper', 'rich'])
        
        # Selecionado Colunas
        df2 = df2.drop(['duration'], axis=1)
        
        return df2
    
    def pre_processamento_2(self, df3):
        
        df3['age'] = self.age_scaler.fit_transform(df3[['age']].values)


        df3['credit amount'] = self.credit_amount_scaler.fit_transform(df3[['credit amount']].values)

        df3['purpose'] = df3['purpose'].apply(lambda x: 'car' if x=='car'
                                      else 'radio/TV' if x=='radio/TV'
                                      else 'furniture/equipment' if x=='furniture/equipment'
                                      else 'others')

        variaveis_categoricas = ['sex','job', 'housing', 'saving accounts', 'checking account',
                                 'purpose','age_cat','year', 'status']

        df3 = pd.get_dummies(df3, columns=variaveis_categoricas, drop_first=True)
        
        return df3
    
    def get_prediction(self, modelo, dados_originais, dados_teste):
        # predições
        predicoes = modelo.predict(dados_teste)
        
        # Juntar predições nos dados originais
        dados_originais['predição'] = predicoes
        
        return dados_originais.to_json(orient = 'records')
