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

 - Estudo dos dados do cliente, assim como da sua situação socioeconômica, é tomada a decisão de conceder ou não o crédito solicitado. Em tese, a análise de crédito deve ser composta por processos muito rigorosos, de modo a proteger a saúde financeira da organização e prevenir prejuízos - mesmo que isso signifique vetar algumas vendas.

- Diversificação de investimentos e a checagem do rating de cada organização com potencial para compor a sua carteira.

 <h2> Objetivo do Case de Estudo </h2>
    
Criar um modelo de **Machine Learning** capaz de fazer a previsão de risco de crédito de clientes de um banco através da análise de dados. 

<h2> Apresentando os Dados </h2>
    
Este projeto estará utilizando o conjunto de dados de um banco alemão que pode ser encontrado nesse [site](https://www.kaggle.com/kabure/german-credit-data-with-risk).
    
    
# 2.0 Solução

## 2.1 Principais Etapas

#### 2.1.1.1 Descrição dos Dados - Estatística Descritiva


**Observações:**

- A média de clientes do banco é de 35 anos.
- O montante de crédito médio dos clientes é de 3236 euros.
- A duração média dos clientes no banco é de 21 meses, sendo que 50% dos clientes tem a duração de 18 meses.


#### 2.1.1.2 Verificando Outliers

