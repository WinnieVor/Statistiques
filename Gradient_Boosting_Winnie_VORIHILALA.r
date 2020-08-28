
install.packages("gbm")
install.packages("caret")
install.packages("xgboost")

install.packages("kernlab") #package où se trouve certains dataset dont spam

install.packages("fastAdaboost") 

library(gbm)
library(caret)
library(xgboost)
library(kernlab)
library(fastAdaboost)

data(spam) #chargement des données spam depuis kernlab

dim(spam)
head(spam)

names(spam) #affiche les noms de colonnes

data <-read.table('/Users/winnievorihilala/Desktop/spambase.data.txt', sep=',', header=TRUE)


dim(data)
class(data)

head(data)

names(data)

summary(data)

lapply(data,class)

datatable <- table(data$type) #fonction table() crée une table de contingence
datatable

library(caret)
set.seed(100)
trainIndex <- createDataPartition(data$type,p=0.8,list=F)
print(length(trainIndex))

#data frame pour les individus en apprentissage
data_train <- data[trainIndex,] 
#data frame pour les individus en test
data_test <- data[-trainIndex,] 
print(dim(data_train))
print(dim(data_test))

nrow(data)
ncol(data)

#set.seed(123)
app <- sample(1:nrow(data), nrow(data)*1/5)
test2_datatest <- data[app,]
test2_dataapp <- data[-app,]
dim(test2_datatest)
dim(test2_dataapp)

model = adaboost(type~., data=data_train,10)
pred_train <- predict(model, data_train)
pred_test <- predict(model, data_test)
cat("L'erreur de précition sur la base apprentissage est égale à :",round(pred_train$error,3))
cat("\nL'erreur de prédiction sur la base test est égale à :",round(pred_test$error,3))

matrice_train <- matrix(ncol=30,nrow=1)
matrice_test <- matrix(ncol=30,nrow=1)
for (i in 1:30){
    model <- adaboost(type~ . , data = data_train, nIter=i)
    pred_train <- predict(model, newdata = data_train)
    pred_test <- predict(model, newdata = data_test)
    matrice_train[i] = pred_train$error
    matrice_test[i] = pred_test$error
    }
x = seq(from = 1, to = 30, by = 1)
plot(x, matrice_train, type="o", col="orange", pch="x", ylim = c(0,0.12), main="Erreur de prédiction en fonction du nombre d'itérations", xlab = "Nombre de prédicteurs", ylab = "Erreur de prédiction")
points(x, matrice_test, col="green", pch="-")
lines(x, matrice_test, col="green")
legend(17,0.12,legend=c("Erreur d'apprentissage","Erreur de test"), col=c("orange","green"),pch=c("x","-"),lty=c(1,2,3), ncol=1)

#Concaténation de data_train et data_test
dataconcat = rbind(data_train,data_test)
dim(dataconcat)

head(dataconcat)

#Fonction qui crée un modèle gbm avec certaines valeurs de paramètres prédéfinies

creation_modele_gbm <- function(gbm_name,n.trees,interaction.depth,shrinkage) {
    set.seed(123) #les données doivent être mélangées avant d'executer gbm
    #train GBM model
    print(system.time(gbm_name <- gbm(
      formula = dataconcat$type~.,
      distribution = "bernoulli",
      data = dataconcat,
      n.trees = n.trees, #NOMBRE D'ITERATIONS 
      interaction.depth = interaction.depth, #PROFONDEUR MAXIMALE DES ARBRES 
      shrinkage = shrinkage, #PARAMETRE DE REGULARISATION A FAIRE VARIER
      train.fraction = 0.8, 
      ))) 
    #print results
    print(gbm_name)
    #affichage graphique de la fonction loss
    n.trees_opt = gbm.perf(gbm_name) #nombre d'itérations optimal
    print(n.trees_opt)
    #Prédiction sur les nouvelles données en utilisant le nombre d'arbres optimal et le type response, 
    #convient plus à la distribution de Bernoulli car transforme chaque résultat en une proba de 0 ou 1
    Yhat_app <- predict(gbm_name, newdata = data_train, n.trees = n.trees_opt, type = "response") #type link par défaut
    Yhat_test <- predict(gbm_name, newdata = data_test, n.trees = n.trees_opt, type = "response") #type link par défaut
    error_app = sum((data_train$type - Yhat_app)^2)/nrow(data_train)
    error_test = sum((data_test$type - Yhat_test)^2)/nrow(data_test)
    cat("Erreur de prédiction sur apprentissage ",error_app)
    cat("\nErreur de prédiction sur test",error_test)
}

creation_modele_gbm(gbm_1,100,1,0.1) #valeurs des paramètres par défaut

creation_modele_gbm(gbm_2,100,2,0.1)

creation_modele_gbm(gbm_3,100,6,0.1) 

creation_modele_gbm(gbm_4,100,1,0.01) 

creation_modele_gbm(gbm_5,100,1,0.9)

creation_modele_gbm(gbm_6,300,1,0.1)

creation_modele_gbm(gbm_7,500,1,0.1)

#Création d'une grille d'hyperparamètres
hyper_grid <- expand.grid(
    n.trees = c(10, 50, 150, 200, 250),
    shrinkage = c(.001, .001, .01, .1, .3),
    interaction.depth = c(1, 2, 3, 4, 5)
)
nrow(hyper_grid)

# grid search 
for(i in 1:nrow(hyper_grid)){
    # reproducibility
    set.seed(123)
  
    # train model
    gbm.tune <- gbm(
        formula = dataconcat$type~.,
        distribution = "bernoulli",
        data = dataconcat,
        n.trees = hyper_grid$n.trees[i], #NOMBRE D ITERATIONS 
        interaction.depth = hyper_grid$interaction.depth[i], #PROFONDEUR DES ARBRES
        shrinkage = hyper_grid$shrinkage[i], #PARAMETRE DE REGULARISATION
        train.fraction = 0.8)

        #add min training error and trees to grid
        hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
        hyper_grid$min_Error[i] <- min(gbm.tune$valid.error)}
    

dim(hyper_grid)

head(hyper_grid)

#affiche la grille hyper_grid avec filtre croissant sur colonne min_Error
head(hyper_grid[order(hyper_grid[,5],decreasing=F), ]) 


#Fonction qui crée un modele gbm optimal

creation_modele_gbm_optimal <- function(gbm_name) {
    set.seed(123) #les donnees doivent etre mélangées avant d'executer gbm
    #train GBM model optimal
    print(system.time(gbm_name <- gbm(
      formula = dataconcat$type~.,
      distribution = "bernoulli",
      data = dataconcat,
      n.trees = 250, #NOMBRE D ITERATIONS 
      interaction.depth = 4, #PROFONDEUR DES ARBRES 
      shrinkage = 0.1, #PARAMETRE DE REGULARISATION 
      train.fraction = 0.8, 
      ))) 
    #print results
    print(gbm_name)
    #affichage graphique de la fonction loss
    n.trees_opt = gbm.perf(gbm_name) #nombre d'itérations optimal
    print(n.trees_opt)
    #Prédiction sur les nouvelles données en utilisant le nombre d'arbres optimal et le type response, 
    #convient plus à la distribution de Bernoulli car transforme chaque résultat en une proba de 0 ou 1
    Yhat_app_opt <- predict(gbm_name, newdata = data_train, n.trees = n.trees_opt, type = "response") #type link par défaut
    Yhat_test_opt <- predict(gbm_name, newdata = data_test, n.trees = n.trees_opt, type = "response") #type link par défaut
    error_app = sum((data_train$type - Yhat_app_opt)^2)/nrow(data_train)
    error_test = sum((data_test$type - Yhat_test_opt)^2)/nrow(data_test)
    cat("Erreur de prédiction sur apprentissage ",error_app)
    cat("\nErreur de prédiction sur test",error_test)
}

creation_modele_gbm_optimal(gbm_optimal_1)

print(system.time(gbm_optimal <- gbm(
      formula = dataconcat$type~.,
      distribution = "bernoulli",
      data = dataconcat,
      n.trees = 250, #NOMBRE D'ITERATIONS 
      interaction.depth = 4, #PROFONDEUR DES ARBRES 
      shrinkage = 0.1, #PARAMETRE DE REGULARISATION 
      train.fraction = 0.8,
      ))) 
    #print results
    print(gbm_optimal)
    #affichage graphique de la fonction loss
    n.trees_opt = gbm.perf(gbm_optimal) #nombre d'itérations optimal
    print(n.trees_opt)
    #Prédiction sur les nouvelles données en utilisant le nombre d'arbres optimal et le type response, 
    #convient plus à la distribution de Bernoulli car transforme chaque résultat en une proba de 0 ou 1    
    Yhat_app_opt <- predict(gbm_optimal, newdata = data_train, n.trees = n.trees_opt, type = "response") #type link par défaut 
    Yhat_test_opt <- predict(gbm_optimal, newdata = data_test, n.trees = n.trees_opt, type = "response") #type link par défaut
    error_app = sum((data_train$type - Yhat_app_opt)^2)/nrow(data_train)
    error_test = sum((data_test$type - Yhat_test_opt)^2)/nrow(data_test)
    cat("Erreur de prédiction sur apprentissage ",error_app)
    cat("\nErreur de prédiction sur test",error_test)

length(Yhat_test_opt) #prediction sur les valeurs de test

round(Yhat_test_opt[1:100],0) #affiche les 100 premières valeurs, nous avons bien des 0 et des 1, plus grosse proportion de 0 que de 1 ce qui est normal

recap <-read.table('/Users/winnievorihilala/Documents/INSA/STAT/tab_recap_2.csv', sep=',', header=TRUE)

recap

summary(gbm_optimal)[1:10,] #renvoie l'importance globale des prédicteurs (leur importance moyenne sur toutes les classes)

