import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#tratamento de dados advindos do arquivo csv
train_data = pd.read_csv("dados/dados_treinamento.csv", header=0, usecols=range(4), names=["Beacon1", "Beacon2","Beacon3","Beacon4"])
train_labels = pd.read_csv("dados/dados_treinamento.csv", header=0, usecols=[4],names=["Quadrante"])
test_data = pd.read_csv("dados/dados_teste.csv", header=0, usecols=range(4), names=["Beacon1", "Beacon2","Beacon3","Beacon4"])
test_labels = pd.read_csv("dados/dados_teste.csv", header=0,usecols=[4], names=["Quadrante"])
#Criação do classificador 
knn = KNeighborsClassifier(n_neighbors=3)
#k é o número de vizinhos que serão considerados para a predição da classe
#Treinamento do classificador 
knn.fit(train_data, train_labels.values.ravel())
# método ravel () transforma os rótulos em array unidimensional

#classificação dos dados de teste 
predicted_labels = knn.predict(test_data)
#Isolamento de rótulos do array de teste
test_labels = test_labels.values.ravel()
#Pritando resultados 
#inicialização da contagem de acertos
matched_labels= 0
for i, prediction in enumerate(predicted_labels):
    if(test_labels[i] == prediction):
        matched_labels +=1
    print(f"Exemplo {i+1}: Quadrante {prediction} x test label - {test_labels[i]}")
porcentagem_de_acerto = (matched_labels/len(predicted_labels))*100
#calculando porcentagem de acerto
print(f'A taxa de acerto foi: {round((porcentagem_de_acerto),2)} %')