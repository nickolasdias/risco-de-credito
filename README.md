# Case de Estudo: Risco de Crédito - Modelo de Machine Learning

# 1.0 Contexto

<h2> O que é o Risco de Crédito ? </h2>

**Risco de Crédito** é a possibilidade de ocorrência de perdas financeiras devido ao não cumprimento das obrigações por parte do tomador. Em outras palavras, é o risco de inadimplência por parte do tomador do crédito. 


<h2> Como o Risco de Crédito funciona? </h2>

Suponhamos que uma pessoa visite uma loja de móveis e eletrodomésticos com o objetivo de comprar uma televisão nova. Lá, ela escolhe o modelo de sua preferência e, ao se dirigir ao caixa, opta pelo pagamento a prazo. O funcionário da loja, então, emite um carnê em nome do cliente, que se compromete a quitar a dívida em até 12 meses. Por fim, o mesmo deixa a loja radiante, levando consigo a tão sonhada televisão.Agora pense um pouco e responda: como a empresa pode ter tanta certeza de que, após entregar o produto ao consumidor, ele realizará mesmo o pagamento?

A resposta é: ela não tem!

E é justamente a essa impossibilidade de garantir um recebimento que se dá o nome de risco de crédito.

Para empresas e investidores, é impossível ignorá-lo.

<h2> Como é feita a classificação dos riscos de crédito ? </h2>

No geral, os riscos de crédito são classificados entre riscos de primeira classe e riscos de segunda classe. Os riscos de primeira classe dizem respeito a operações de crédito com alta chance de inadimplência. Por conta disso, exigem um volume maior de garantias para proteger o credor. Já os riscos de segunda classe estão relacionados a operações de risco menor. Em geral, o bem objeto da transação pode ser revisto em caso de não-pagamento, o que garante que o credor possa comercializá-lo novamente para cobrir parte da dívida.

<h2> Quais são as principais táticas utilizadas para minimizar o risco de crédito? </h2>

 - Estudo dos dados do cliente, assim como da sua situação socioeconômica para à tomada de decisão de conceder ou não o crédito solicitado. Em tese, a análise de crédito deve ser composta por processos muito rigorosos, de modo a proteger a saúde financeira da organização e prevenir prejuízos - mesmo que isso signifique vetar algumas vendas.

- Diversificação de investimentos e a checagem do rating de cada organização com potencial para compor a sua carteira.

 <h2> Objetivo do Case de Estudo </h2>
    
Criar um modelo de **Machine Learning** capaz de fazer a previsão de risco de crédito "bom" ou "ruim" de clientes de um banco através da análise de dados. 

<h2> Apresentando os Dados </h2>
    
Este projeto utilizou o conjunto de dados de um banco alemão que pode ser encontrado nesse [site](https://www.kaggle.com/kabure/german-credit-data-with-risk). Neste conjunto de dados há:

- 1000 linhas - caracterizado por cada cliente.
- 11 colunas - sendo 10 que explicam o resultado da variável resposta.

**Observação:** Antes de fazer qualquer análise e modelagem, separei o conjunto de dados entre **treino** e **teste**. 

<h2> Dicionário de Dados </h2>

- **Age:** Idade.

- **Sex:** Gênero.

- **Job:** Emprego - 0 - não qualificados e não residentes, 1 - não qualificados e residentes, 2 - qualificados, 3 - altamente qualificados.

- **Housing:** (text: own, rent, or free) - (própria, alugado ou gratuito).

- **Saving Account:** Conta poupança (text - little, moderate, quite rich, rich).

- **Checking Account:** Conta corrente.

- **Credit Amount:** Montante de Crédito.

- **Duration:** Duração de tempo, em meses, dos clientes.

- **Purpose:** Objetivo do Crédito (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others).

- **Risk:** Risco de Crédito se é bom ou ruim.
    
    
# 2.0 Solução

Nesta seção abordarei as principais etapas que foram importantes para a construção do modelo de machine learning como: Estatística Descritiva, Análise Explortória de Dados, Pré-Processamento, Seleção de Variáveis, Modelos de Machine Learning, Hipertunagem de Parâmetros, Performance dos Modelos no Conjunto de Teste e a Avaliação do Modelo.

## 2.1 Principais Etapas

### 2.1.1 Descrição dos Dados - Estatística Descritiva

É através da estatística descritiva que conseguimos obter um resumo dos dados, pois ajuda a sintetizar os dados de maneira direta, preocupando-se com menos variações e intervalos de confiança. Portanto, nessa etapa, o que fiz é entender as características dos dados por meio desta análise para ter uma noção melhor do contexto que estou modelando.

Desta forma, existem duas métricas em que tem-se de manter os olhos: a tendência central e a distribuição dos dados.

- **Tendência Central:** são estatísticas como mediana, média, quartis, valor máximo e valor mínimo.

- **Distribuição dos Dados:** é o comportamento dos dados em torno da média e mediana .

Portanto, realizei a estatística descritiva das **variáveis numéricas e variáveis categóricas**.

**Variáveis Numéricas**

![001](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/001.png)

**Observações:**

- A média de clientes do banco é de 35 anos.
- O montante de crédito médio dos clientes é de 3236 euros.
- A duração média dos clientes no banco é de 21 meses, sendo que 50% dos clientes tem a duração de 18 meses.


##### 2.1.1.1 Verificando Outliers

![002](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/002.png)

![003](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/003.png)

**Observações:**

- Pode-se verificar que a variável `age` tem outliers acima de 62 anos.
- Talvez não se necessário tratá-los pelo fato da distribuição ser assimétrica e não ter tantos dados fora da curva.

![004](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/004.png)

![005](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/005.png)

**Observações:**

- Pode-se observar que existem valores de créditos dos clientes acima de 7909 euros, que são considerados outliers.

**Variáveis Categóricas**

![006](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/006.png)

**Observações:**

- Cerca de 486 (69%) dos clientes do banco são do gêneno masculino.
- Cerca de 448 (64%) dos clientes do banco têm empregos qualificados.
- Cerca de 504 (72%) dos clientes possuem habitação própria.
- Cerca de 428 (74%) dos clientes possuem uma conta poupança do tipo pequena.
- Cerca de 201 (47%) dos clientes possuem uma conta corrente do tipo pequena. 
- A maioria dos clientes(17.7%) duram 24 meses no banco.
- Cerca de 230 (33%) dos clientes possuem o objetivo de comprar um carro.
- Cerca de 490 (70%) dos clientes possuem risco de crédito bom.

Analisando alguns resultados acima, a maioria dos clientes possuem risco de crédito bom (70%). E isso acontece, pelo fato que a maioria dos clientes possuem empregos qualificados (64%) e habitação própria (72%). Logo, verificarei através da **Análise Exploratória de Dados** as variáveis que explicam a **variável resposta**.

### 2.1.2 Análise Exploratória de Dados

Em resumo, a importância da **Análise Exploratória de Dados** é compreender e medir forças de como as variáveis impactam no fenômeno que se está modelando. Dessa forma, os objetivos da AED são:

- Ganhar experiência de negócio.

- Entender quais variáveis são importantes para o contexto modelado.

Para realizar a Análise Exploratória de Dados, utilizei a biblioteca **Sweetviz** que fornece uma rápida e eficiente análise dos dados. Logo, os gráficos encontram-se no [projeto](https://github.com/nickolasdias/risco-de-credito/blob/main/notebook/risco_de_credito.ipynb). Portanto, só considerei para este **readme**, as observações feitas.

**Observações Relevantes:**

- Clientes do gênero masculino recebem um maior risco de crédito bom do que clientes do gênero feminino.
- Clientes que tem empregos qualificados recebem mais risco de crédito bom do que clientes que tem apenas um emprego que não é qualificado e também empregos que são altamente qualificados.
- Clientes que tem casa própria recebem mais risco de crédito bom do que clientes que tem casas alugadas ou com o termo 'free'.
- Clientes com conta poupança do tipo rica recebem um maior risco de crédito bom do que clientes com conta poupança muito rica, moderada ou pequena. 
- Clientes com conta corrente do tipo rica recebem um maior risco de crédito bom do que clientes com conta corrente do tipo moderada ou pequena.
- Clientes com crédito de 0 até 5000 euros rebecem um maior risco de crédito bom do que clientes com crédito de 5000 a 10000 euros.
- Clientes com até 25 meses de banco recebem um maior risco de crédito bom do que clientes com mais de 25 meses de banco. Isto é, quanto mais novo é o cliente, mais ele recebe um risco de crédito bom.
- Clientes que tem o objetivo de comprar radio/TV recebem um maior risco de crédito bom do que clientes que tem o objetivo de comprar um carro, mobiliaria/equipamentos e etc.

**Associações Relevantes com a Variável Target:**
- `age` e `risk`: 0.10
- `credit amount`e `risk`: 0.16
- `duration` e `risk`: 0.22

Verifiquei que as variáveis `age`, `credit amount` e `duration` têm as correlações mais significativas com a variável resposta. Logo, são variáveis que  têm um peso mais significativo para determinar o risco de crédito do cliente. Portanto, farão parte da construção do modelo.

### 2.1.3 Pré-Processamento

Na etapa de pré-processamento, fiz a transformação dos dados para o preparo de treino dos modelos de **Machine Learning**. A razão para isto é que a aprendizagem dos algoritmos é facilitada com dados numéricos. Portanto, realizei as etapas:

- Tratando Dados Faltantes
- Featuring Engineering
- Filtragem de Variáveis
- Preparação dos Dados

Todas essas etapas, podem ser verificadas no [notebook](https://github.com/nickolasdias/risco-de-credito/blob/main/notebook/risco_de_credito.ipynb).

### 2.1.4 Seleção de Variáveis

Já a seleção de variáveis é importante para facilitar a compreensão dos algoritmos de **Machine Learning**. Esse passo verifiquei quais variáveis do conjunto de dados são colineares, ou seja, variáveis que explicam a mesma informação. Portanto, é preciso remover essas variáveis.

Para fazer a seleção de variáveis utilizei o algoritmo **Decision Tree Classifier** que calcula quais as variáveis são mais importantes para o modelo.

![012](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/012.png)

**Observação:**

- As variáveis que o algoritmo **Decision Classifier** selecionou como sendo as mais importantes são `credit amount` e `age`. 
- Temos features que não tem tanta importância, mas selecionei todas as variáveis para compor o modelo.

### 2.1.5 Machine Learning

Neste módulo, apliquei os modelos de **Machine Learning** em que a motivação para utilizar estes algoritmos é construir um modelo inteligente capaz de classificar se cada cliente vai ter um **risco de crédito** bom ou ruim. Para isso, utilizei algoritmos de aprendizado de máquina supervisionado como:

- Decision Tree
- KNeighbors Classifier (KNN)
- Random Forest Classifier

Nesses algoritmos diferentes parâmetros foram testados para diversos resultados.

![007](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/007.png)

**Observações:**

- Analisando os resultados e pensando no contexto de negócio, o modelo preferível é que tenha um bom **F1-Score**, pois o que me interessa é ter o resultado dessa métrica mais significativa, nas quais a **Precisão** e a **Sensibilidade** sejam aumentadas e equilibradas pela média harmonica.

Portanto, escolhei alguns modelos para serem avaliado no conjunto de teste.

**Modelos Escolhidos:**

- Decision Tree Classifier(max_depth=3)
- Random Forest Classifier(max_depth=7)
- Random Forest Classifier(max_depth=10)

#### 2.1.5.1 Balanceando os Dados

Para verificar a melhora do desempenho dos modelos, balanciei os dados, na qual é usada uma técnica que cria dados sintéticos com a biblioteca [**SMOTE**](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/).


![008](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/008.png)

**Observação:**

- As classes da variável target foram balanceadas com as classes tendo a mesma frequência.

#### 2.1.5.2 Machine Learning Dados Balanceados

![009](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/009.png)

**Observação:**

- Os dados balanceados apresentaram uma melhora significativa nos resultados dos modelos. Logo, o modelo **Random Forest Classifier(max_depth=15)** apresentou uma **Acurácia**, **Precisão**, **Sensibilidade** e **F1-Score** melhor que os demais modelos.

Portanto, escolhi o modelo **Random Forest Classifier(max_depth=15)** para ser avaliado nos dados de teste.

**Modelo Escolhido:**

- Random Forest Classifier(max_depth=15)

### 2.1.6 Hipertunagem de Parâmetros

Em machine Learning, a hipertunagem de parâmetros é a escolha de um conjunto de hiperparâmetros ótimos para um algoritmo, cujo valores são utilizados para controlar o processo de aprendizagem. Em contraste, os valores de outros parâmetros (tipicamente pesos de nós) são aprendidos.

Portanto, utilizei a técnica de [**RandomSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), pois o método aqui descrito é um tipo de pesquisa aleatória local em que cada iteração depende da solução candidata da iteração anterior.

#### 2.1.6.1 Comparando Modelo Base x Modelo Hipertunado no Conjunto de Treino

**Dados Não-Balanceados**

![010](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/010.png)

**Observações:**

- Analisando as métricas **F1-Score** e **Acurácia**,  os modelos **Decision Tree Classifier(max_depth=1)** e **Random Forest Classifier(max_depth=1)** são que tem desempenhos melhores. Porém, testei todos esses modelos no conjunto de teste.

**Dados Balanceados**

![011](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/011.png)

**Observações:**

- O modelo base **Random Forest Classifier(max_depth=19)** apresentou um desempenho melhor do que o modelo hipertunado **Random Forest Classifier(max_depth=15)**. Logo, esses modelos treinados com dados balanceados apresentaram performance melhor do que os modelos treinados com dados não balanceados. Porém, verifiquei se essas performances continuam boas no conjunto de teste.

### 2.1.7 Conjunto de Teste

#### 2.1.7.1 Performance dos Modelos no Conjunto de Teste

**Dados Balanceados**

![013](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/013.png)

![014](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/014.png)

**Observações:**

- Analisando as performances dos modelos nos dados de teste, verifiquei que o desempenho nos dados de testes está abaixo. Portanto, está acontecendo o processo de **Overfiting** no modelo.

**Dados Não-Balanceados**

![015](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/015.png)

![016](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/016.png)

![017](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/017.png)

![018](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/018.png)

![019](https://github.com/nickolasdias/risco-de-credito/blob/main/imagens/019.png)

**Observações:**

- Analisando o desempenho de todos os modelos, o algoritmo **Random Forest Classifier(max_depth=1)** e **Decision Tree Classifier(max_depth=1)** apresentaram ter as melhores performances, levando em consideração as métricas **Acurácia** (70%), **Precisão** (70%), **Sensibilidade** (100%) e **F1-Score** (82.35%). Logo, escolhi o modelo de Random Forest Classifier.

#### 2.1.7.2 Avaliação do Modelo

Considerando as métricas do modelo **Random Forest Classifier(max_depth=1)**, posso dizer que o modelo acerta com precisão 7 a cada 10 clientes, se terá um risco de crédito bom ou ruim. Enquanto que a probabilidade de o cliente ter, de fato, um risco de crédito bom é de 100%, que é sua sensibilidade. Logo, a combinação da precisão com a sensibilidade pela média harmônica é de 82.35%, na qual para a quantidade de dados utilizados no conjunto de dados é um desempenho do modelo satisfatório, nesse primeiro momento.

Logo, para melhorar a performance o modelo e equilibrar mais as métricas entre precisão e sensibilidade é necessário a utilização de mais dados para que o mesmo tenha uma aprendizagem melhor. Não foram utilizados mais parâmetros no modelo para não aumentar sua complexidade, pois o algoritmo de árvore tem uma facilidade de overfitar o modelo conforme sua quantidade de parâmetros. 

# 3.0 Deploy do Modelo para Produção

Nesse projeto, foi realizado o deploy do melhor modelo em produção em **Local Host** e no [**Heroku**](https://id.heroku.com/login) que pode ser encontrado notebook do projeto.

# 4.0 Próximos Passos

- Coletar mais dados para melhorar o modelo
- Realizar mais seções do CRISP.
- Experimentar novos modelos de machine learning para obter uma performance melhor.

# Referências

- https://www.pwc.com.br/pt/consultoria-negocios/gestao-risco-compliance/risco-de-credito.html
- https://maisretorno.com/portal/termos/r/risco-de-credito
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- https://medium.com/data-hackers/machine-learning-para-avalia%C3%A7%C3%A3o-de-risco-de-cr%C3%A9dito-49578b03b4b8
- https://pypi.org/project/sweetviz/
