
#Installation
install.packages("kernlab")
install.packages("mlbench")
install.packages("rpart")
install.packages("randomForest")
install.packages("rpart.plot")

#Chargement
library(kernlab)
library(mlbench)
library(rpart)
library(randomForest)
library(rpart.plot)

data(spam) #charge les données spam présents dans kernlab
dim(spam) #affiche les dimensions de spam
class(spam) #affiche le type de spam
head(spam) #affiche les 6 premières lignes et toutes les colonnes de spam

#Descriptif du jeu de données 
?spam 

#Vérification des types de chacune des variables
#lapply(spam,class) // décommenter

#Création d'une table de contingence relative à la variable catégorielle type
set.seed(9146301)
ytable <- table(spam$type) #fonction table() crée une table de contingence
ytable

#Split des données en apprentissage et test
app <- c(sample(1:ytable[2], ytable[2]/2), sample((ytable[2] + 1):nrow(spam),ytable[1]/2)) #indices
spam.app <- spam[app, ]
spam.test <- spam[-app, ]
dim(spam.app)
dim(spam.test)

#Création d'une table de contigence relative à la variable catégorielle spam.app$type
apptable <- table(spam.app$type)
apptable

#Création d'une table de contigence relative à la variable catégorielle spam.test$type
testtable <- table(spam.test$type)
testtable

# Récupération du nombre de lignes et colonnes de spam.app qui serviront plus bas
n <- nrow(spam.app)
p <- ncol(spam.app) - 1
paste(n)
paste(p)

library(rpart)

t_def = rpart(type ~ .,data = spam.app)

#Affichage de l'arbre par défaut créé avec rpart 
t_def

# Affichage graphique de l'arbre
plot(t_def)
text(t_def, cex = 0.5)

#summary(t_def)

tstump <- rpart(type ~ ., spam.app, control = rpart.control(maxdepth = 1))

tstump

plot(tstump)
text(tstump)

summary(tstump)

tmax <- rpart(type ~ ., data = spam.app, control = rpart.control(minsplit = 1,cp = 0))

#tmax

#affichage graphique
plot(tmax)
text(tmax, cex = 0.5)

plotcp(tmax)

tprune <- prune(tmax, cp = tmax$cptable[which.min(tmax$cptable[, 4]), 1]) #cp = cp qui mimimise xerror
#tprune

head(tmax$cptable)

tmax$cptable[which.min(tmax$cptable[, 4]), 1] #cp optimal

plot(tprune)
text(tprune, cex = 0.8)

prp(tprune,extra=1)

thres1SE <- sum(tmax$cptable[ which.min(tmax$cptable[, 4]), 4:5]) #seuil d'élagage
cp1SE <- tmax$cptable[ min(which(tmax$cptable[, 4] <= thres1SE)),1] #on prend le cp qui minimise xerror et qui est <= au seuil
tprune_1se <- prune(tmax, cp = cp1SE)
print(thres1SE)
print(cp1SE)

plot(tprune_1se)
text(tprune_1se, cex = 0.8)

identical(t_def, tprune_1sd)
identical(tprune, tprune_1sd)

identical(t_def, tmax)
identical(tmax, tprune)
identical(tstump, tprune)
identical(tstump, tmax)
identical(tprune, tprune_1sd)

#arbre stump, profondeur=1, 2 feuilles
predstump <- predict(tstump, spam.test, type="class")
errstump <- round(sum(predstump!=spam.test$type)/nrow(spam.test), 3)
appstump <- round(sum(predict(tstump, spam.app, type="class")!=spam.app$type)/nrow(spam.app), 3)

#arbre maximal
predmax <- predict(tmax, spam.test, type="class")
errmax <- round(sum(predmax!=spam.test$type)/nrow(spam.test), 3)
appmax <- round(sum(predict(tmax, spam.app, type="class")!=spam.app$type)/nrow(spam.app), 3)

#arbre 1 SE
pred_1se<- predict(tprune_1se, spam.test, type="class")
err_1se <- round(sum(pred_1se!=spam.test$type)/nrow(spam.test), 3)
app_1se <- round(sum(predict(tprune_1se, spam.app, type="class")!=spam.app$type)/nrow(spam.app), 3)


#arbre otpimal
predprune <- predict(tprune, spam.test, type="class")
errprune <- round(sum(predprune!=spam.test$type)/nrow(spam.test), 3)
appprune <- round(sum(predict(tprune, spam.app, type="class")!=spam.app$type)/nrow(spam.app), 3)

cat("L\'erreur de prédiction de l\'arbre de profondeur 1 sur le jeu de données de test est:",errstump,"\n")
cat("L\'erreur de prédiction de l\'arbre de profondeur 1 sur le jeu de données d'apprentissage est:",appstump,"\n")
print("-------")

cat("L\'erreur de prédiction de l\'arbre de profondeur maximale sur le jeu de données de test est:",errmax,"\n")
cat("L\'erreur de prédiction de l\'arbre de profondeur maximale sur le jeu de données d'apprentissage est:",appmax,"\n")
print("-------")

cat("L\'erreur de prédiction de l\'arbre 1SE sur le jeu de données de test est:",err_1se,"\n")
cat("L\'erreur de prédiction de l\'arbre 1SE sur le jeu de données d'apprentissage est:",app_1se,"\n")
print("-------")

print("-------")
cat("L\'erreur de prédiction de l\'arbre optimal sur le jeu de données de test est:",errprune,"\n")
cat("L\'erreur de prédiction de de l\'arbre optimal sur le jeu de données d'apprentissage est:",appprune,"\n")
print("-------")

data(Ozone)
dim(Ozone) #affiche les dimensions de Ozone
class(Ozone) #affiche le type de Ozone
head(Ozone) 

#Descriptif du jeu de données 
#?Ozone 

lapply(Ozone,class)

#conversion des factor en numeric
Ozone$V1 <- as.numeric(Ozone$V1)
Ozone$V2 <- as.numeric(Ozone$V2)
Ozone$V2 <- as.numeric(Ozone$V2)

summary(Ozone)

#suppression des valeurs NA
Ozone <- na.omit(Ozone)                            

summary(Ozone)

boxplot(Ozone)

summary(Ozone$V4)

#Split des données en test et en apprentissage

set.seed(111)
app <- sample(1:nrow(Ozone), nrow(Ozone)*1/5) #80% apprentissage et 20% test
Ozone_app <- Ozone[app,]
Ozone_test <- Ozone[-app,]

#Vérification des dimensions de app et test

dim(Ozone)
dim(Ozone_app)
dim(Ozone_test)

#Création de l'arbre par défaut

o_def=rpart(V4 ~ .,data = Ozone_app)
plot(o_def)
text(o_def, cex = 0.5)

o_def

#summary(o_def)

#Création de l'arbre stump à 2 feuilles et de profondeur 1

ostump <- rpart(V4 ~ ., Ozone_app, control = rpart.control(maxdepth = 1))

plot(ostump)
text(ostump)

summary(ostump)

# Création de l'arbre maximale

omax <- rpart(V4 ~ ., data = Ozone_app, control = rpart.control(cp = 0))

omax

#summary(omax)

plot(omax)
text(omax, cex = 0.5)

omax$cptable

printcp(omax)

plotcp(omax)

min(omax$cptable[, 4])

#Création de l'arbre optimal

oprune <- prune(omax, cp = omax$cptable[which.min(omax$cptable[, 4]), 1])
plot(oprune)
text(oprune, cex = 0.8)

#Création de l'arbre 1SE

othres1SE <- sum(omax$cptable[ which.min(omax$cptable[, 4]), 4:5])
ocp1SE <- omax$cptable[ min(which(omax$cptable[, 4] <= othres1SE)), 1]
oprune_1se <- prune(omax, cp = ocp1SE)
plot(oprune_1se)
text(oprune_1se, cex = 0.8)

identical(o_def, omax)
identical(omax, oprune)
identical(o_def, oprune)
identical(o_def, oprune_1se)
#identical(o_def, oprune)

predstump_o <- predict(ostump, Ozone_test)
errstump_o <- round(sum((predstump_o-Ozone_test$V4)^2)/nrow(Ozone_test), 4)
predstump_o_a <- predict(ostump, Ozone_app)
appstump_o <- round(sum((predstump_o_a-Ozone_app$V4)^2)/nrow(Ozone_app), 4)

predmax_o <- predict(omax, Ozone_test)
errmax_o <- round(sum((predmax_o-Ozone_test$V4)^2)/nrow(Ozone_test), 4)
predmax_o_a <- predict(omax, Ozone_app)
appmax_o <- round(sum((predmax_o_a-Ozone_app$V4)^2)/nrow(Ozone_app), 4)

pred1SE_o <- predict(oprune_1se, Ozone_test)
err1SE_o <- round(sum((pred1SE_o-Ozone_test$V4)^2)/nrow(Ozone_test), 4)
pred1SE_o_a <- predict(oprune_1se, Ozone_app)
app1SE_o <- round(sum((pred1SE_o_a-Ozone_app$V4)^2)/nrow(Ozone_app), 4)

predprune_o <- predict(oprune, Ozone_test)
errprune_o <- round(sum((predprune_o-Ozone_test$V4)^2)/nrow(Ozone_test), 4)
predprune_o_a <- predict(oprune, Ozone_app)
appprune_o <- round(sum((predprune_o_a-Ozone_app$V4)^2)/nrow(Ozone_app), 4)

cat("La MSE de l\'arbre de profondeur 1 sur le jeu de données de test est:",errstump_o,"\n")
cat("La MSE de l\'arbre de profondeur 1 sur le jeu de données d'apprentissage est:",appstump_o,"\n")
print("-------")

cat("La MSE de l\'arbre de profondeur maximale sur le jeu de données de test est:",errmax_o,"\n")
cat("La MSE de l\'arbre de profondeur maximale sur le jeu de données d'apprentissage est:",appmax_o,"\n")
print("-------")

cat("La MSE de l\'arbre 1SE sur le jeu de données de test est:",err1SE_o,"\n")
cat("La MSE de l\'arbre 1SE sur le jeu de données d'apprentissage est:",app1SE_o,"\n")
print("-------")

cat("La MSE de l\'arbre de profondeur optimale sur le jeu de données de test est:",errprune_o,"\n")
cat("La MSE de l\'arbre de profondeur optimale sur le jeu de données d'apprentissage est:",appprune_o,"\n")
print("-------")

rmse_err1SE_o <- sqrt(err1SE_o)
rmse_app1SE_o <- sqrt(app1SE_o)

rmse_errstump_o <- sqrt(errstump_o)
rmse_appstump_o <- sqrt(appstump_o)

rmse_errmax_o <- sqrt(errmax_o)
rmse_appmax_o <- sqrt(appmax_o)

rmse_errprune_o <- sqrt(errprune_o)
rmse_appprune_o <- sqrt(appprune_o)

cat("La RMSE de l\'arbre 1SE sur le jeu de données de test est:",rmse_err1SE_o,"\n")
cat("La RMSE de l\'arbre 1SE sur le jeu de données d'apprentissage est:",rmse_app1SE_o,"\n")
print("-------")
cat("La RMSE de l\'arbre de profondeur 1 sur le jeu de données de test est:",rmse_errstump_o,"\n")
cat("La RMSE de l\'arbre de profondeur 1 sur le jeu de données d'apprentissage est:",rmse_appstump_o,"\n")
print("-------")
cat("La RMSE de l\'arbre de profondeur maximale sur le jeu de données de test est:",rmse_errmax_o,"\n")
cat("La RMSE de l\'arbre de profondeur maximale sur le jeu de données d'apprentissage est:",rmse_appmax_o,"\n")
print("-------")
cat("La RMSE de l\'arbre de profondeur optimale sur le jeu de données de test est:",rmse_errprune_o,"\n")
cat("La RMSE de l\'arbre de profondeur optimale sur le jeu de données d'apprentissage est:",rmse_appprune_o,"\n")
print("-------")

vozone <- VSURF(V4 ~ ., data = Ozone, na.action = na.omit)

summary(vozone)

plot(vozone, step = "thres", imp.sd = FALSE, var.names = TRUE)

number <- c(1:3, 5:13)
number[vozone$varselect.thres]

number[vozone$varselect.interp]

library(randomForest)
## ----data---------------------------------------------------------------

library(kernlab)

data(spam)
?spam

dim(spam)

spam[1:5, 1:5]

set.seed(9146301)
ytable <- table(spam$type)
app <- c(sample(1:ytable[2], ytable[2]/2), sample((ytable[2] + 1):nrow(spam),
ytable[1]/2))
spam.app <- spam[app, ]
table(spam.app$type)

n <- nrow(spam.app)
p <- ncol(spam.app) - 1
spam.test <- spam[-app, ]
table(spam.test$type)

## ----bag-----------------------------------------------------------------

bag <- randomForest(type ~ ., data = spam.app, mtry = p)
bag
## ----bag_err-------------------------------------------------------------

predbag <- predict(bag, spam.test)
errbag <- round(sum(predbag != spam.test$type)/nrow(spam.test), 3)
errbag

plot(bag)

tail(bag$err.rate)

## ----rf------------------------------------------------------------------

rf <- randomForest(type ~ ., spam.app)
rf
errrf

plot(rf)

## ----rf_err--------------------------------------------------------------

predrf <- predict(rf, spam.test)
errrf <- round(sum(predrf != spam.test$type)/nrow(spam.test), 3)
errrf

rf1

rfpsur3

rfp

## ----iv------------------------------------------------------------------

rf <- randomForest(type ~ ., data = spam.app, ntree = 1000, do.trace = 100)

rf <- randomForest(type ~ ., data = spam.app, importance = TRUE)
nlev <- nlevels(spam.app$type)
rfimp <- rf$importance[, nlev + 1]
barplot(rfimp)

#rf$importance

varImpPlot(bagstump)

varImpPlot(rfstump)

## ----sort_iv-----------------------------------------------------------

rfimpsort <- sort(rfimp, decreasing = TRUE, index.return = TRUE)
barplot(rfimpsort$x)

colnames(spam.app[rfimpsort$ix[1:8]])

plot(tstump)
text(tstump)

errstump

bagstump <- randomForest(type~., spam.app, maxnodes=2, mtry=p, importance=TRUE) #foret stump avec bagging non élagué
bagstumpimpsort <- sort(bagstump$importance[, nlev+1], decreasing=TRUE,
index.return=TRUE)
barplot(bagstumpimpsort$x)

colnames(spam.app[bagstumpimpsort$ix[1:8]])

predbagstump <- predict(bagstump, spam.test)
errbagstump <- round(sum(predbagstump!=spam.test$type)/nrow(spam.test), 3)
errbagstump

rfstump <- randomForest(type~., spam.app, maxnodes=2, importance=TRUE)
rfstumpimpsort <- sort(rfstump$importance[, nlev+1], decreasing=TRUE,
index.return=TRUE)
barplot(rfstumpimpsort$x)

colnames(spam.app[rfstumpimpsort$ix[1:8]])

predrfstump <- predict(rfstump, spam.test)
errrfstump <- round(sum(predrfstump!=spam.test$type)/nrow(spam.test), 3)
errrfstump

#Influence mtry

rf1 <- randomForest(type~., spam.app, mtry=1, importance=TRUE) #mtry =1 (rf1)
tail(rf1$err.rate[, 1], 1)

rf1impsort <- sort(rf1$importance[, nlev+1], decreasing=TRUE, index.return=TRUE)
barplot(rf1impsort$x)

colnames(spam.app[rf1impsort$ix[1:8]])

barplot(rfimpsort$x)

colnames(spam.app[rfimpsort$ix[1:8]])

tail(rf$err.rate[, 1], 1)

rfpsur3 <- randomForest(type~., spam.app, mtry=p/3, importance=TRUE) #mtry =p/3 (rfpsur3)
tail(rfpsur3$err.rate[, 1], 1)

rfpsur3impsort <- sort(rfpsur3$importance[, nlev+1], decreasing=TRUE,
index.return=TRUE)
barplot(rfpsur3impsort$x)

colnames(spam.app[rfpsur3impsort$ix[1:8]])

rfp <- randomForest(type~., spam.app, mtry=p, importance=TRUE) #mtry = p (rfp)
tail(rfp$err.rate[,1], 1)

rfpimpsort <- sort(rfp$importance[,nlev+1], decreasing=TRUE, index.return=TRUE)
barplot(rfpimpsort$x)

colnames(spam.app[rfpimpsort$ix[1:8]])

vimultntree100 <- matrix(NA, nrow = 20, ncol = p)
for (i in 1:20) {
rf <- randomForest(type ~ ., spam.app, ntree = 100, importance = TRUE)
vimultntree100[i, ] <- rf$importance[, nlev + 1]
}
boxplot(vimultntree100)

vimult <- matrix(NA, nrow = 20, ncol = p)
for (i in 1:20) {
rf <- randomForest(type ~ ., spam.app, importance = TRUE)
vimult[i, ] <- rf$importance[, nlev + 1]
}
boxplot(vimult)



sort(round(importance(rf), 2)[,1])

sort(round(importance(rfstump), 2)[,1])

install.packages("VSURF")

## ----vsurf_load----------------------------------------------------------

library(VSURF)


## ----vsurf_spam----------------------------------------------------------

small.n <- 500
ytable.app <- table(spam.app$type)
small.app <- c(sample(1:ytable.app[2], ytable.app[2]/n * small.n),
sample((ytable.app[2] + 1):n, ytable.app[1]/n * small.n))
spam.small.app <- spam.app[small.app, ]
table(spam.small.app$type)


## ----vsurf_spam_run-----------------------------------------------------
# BE AWARE THAT THE FOLLOWING COMMAND TAKES AROUND 1 HOUR TO RUN
# ON A COMPUTER, WITH PARALLEL COMPUTING USING 3 CORES

vsurf.spam <- VSURF(type~., spam.app, parallel = TRUE, ncores = 3)

plot(vsurf.spam, cex.axis = 1.1, cex.lab = 1.2)
colnames(spam.app[vsurf.spam$varselect.interp])


## ------------------------------------------------------------------------

summary(vsurf.spam)

colnames(spam.app[vsurf.spam$varselect.interp])
colnames(spam.app[vsurf.spam$varselect.pred])

vsurf.stump <- VSURF(type ~ ., spam.small.app, maxnodes = 2)

summary(vsurf.stump)

plot(vsurf.stump)

colnames(spam.app[vsurf.stump$varselect.interp])

colnames(spam.app[vsurf.stump$varselect.pred])


