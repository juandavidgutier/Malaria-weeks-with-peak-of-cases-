###THIS SCRIPT IS TO RUN THE MODEL OF MACHINE LEARNING OF THE PAPER "Artificial Intelligence to predict weeks with peak of cases of malaria in Colombia: An implementation of XGBoost method"
library(data.table)
library(astsa)
library(dplyr)
library(xgboost)
library(caret)
library(pROC)
library(tidyverse)
library(dismo)
library(gbm)
library(iml)
library(ggplot2)
options(scipen=999)

#Parallel Processing
library(doParallel)
cl <- makeCluster(2)
registerDoParallel(cl)


setwd("D:/jd/clases/UDES/articulo malaria total/ai/")


#######
#######ElBagre
muni<- read.csv("ElBagre.csv", sep = ",")
munici_ElBagre <- round(muni, 2)
str(munici_ElBagre)

munici_ElBagre$Peak.of.cases <- as.factor(munici_ElBagre$Peak.of.cases)
levels(munici_ElBagre$Peak.of.cases) = c("No.Peak", "Peak")

summary(munici_ElBagre)

#Since the response variable is a binary categorical variable, you need to make sure the training data has approximately equal proportion of classes.
table(munici_ElBagre$Peak.of.cases)
'%ni%' <- Negate('%in%')  # define 'not in' func

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici_ElBagre$Peak.of.cases, p=0.7, list = F)  # 70% training data
trainData <- munici_ElBagre[trainDataIndex, ]
table(trainData$Peak.of.cases)
testData <- munici_ElBagre[-trainDataIndex, ]
table(testData$Peak.of.cases)


# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)


#  model with caret
set.seed(99)  # for reproducibility
xgb_trcontrol$sampling <- "smote"
xgboost_model = caret::train(Peak.of.cases ~ ., data = trainData, 
                      method = "xgbTree",
                      metric = "Sensitivity",
                      prob.model = TRUE, na.action = na.omit, 
                      trControl = xgb_trcontrol,
                      tuneGrid = xgbGrid)

# performance metrics
gbm.pred_ElBagre <- predict(xgboost_model, testData, na.action = na.pass)
ConfMat_ElBagre <- caret::confusionMatrix(gbm.pred_ElBagre, testData$Peak.of.cases, positive = "Peak")
print(ConfMat_ElBagre)


###Loop for ROC AUC of each fold
fold <- kfold(trainData, k=10)
auc<-rep(NA,10)
for (i in 1:10){
  training_fold <- trainData[fold == i, ]
  test_fold <- trainData[fold != i, ]
  y_pred <- predict(xgboost_model, newdata = test_fold, type="prob", na.action = na.pass)
  auc[i] <- result.roc <- roc(test_fold$Peak.of.cases, y_pred$Peak, direction = "<")$auc
}
aucXgboostk10_ElBagre <- auc
aucXgboostk10Mean_ELBagre <- mean(auc)


###### Interpretation
X <- munici_ElBagre%>%
  dplyr::select(- Peak.of.cases) %>%
  as.data.frame()

# Permutation Feature Importance
predictor_imp <- Predictor$new(xgboost_model, data = X, y = munici_ElBagre$Peak.of.cases, type = "prob")
importace_rf_ElBagre = FeatureImp$new(predictor_imp, loss = "ce", compare = 'difference', n.repetitions = 500)
plot(importace_rf_ElBagre) + theme_bw()


# Plot feature interactions
predictor_inter <- Predictor$new(xgboost_model, data = X, type = "prob")
interactions_ElBagre <- Interaction$new(predictor_inter)
interactions_ElBagre$plot()

# Shapley values 
predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre57 <- Shapley$new(predictor_shapley, x.interest = X[57, ]) # 57th observation
plot(shapley_rf_ElBagre57) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre77 <- Shapley$new(predictor_shapley, x.interest = X[77, ]) # 77th observation
plot(shapley_rf_ElBagre77) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre93 <- Shapley$new(predictor_shapley, x.interest = X[93, ]) # 93th observation
plot(shapley_rf_ElBagre93) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre108 <- Shapley$new(predictor_shapley, x.interest = X[108, ]) # 108th observation
plot(shapley_rf_ElBagre108)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre145 <- Shapley$new(predictor_shapley, x.interest = X[145, ]) # 145th observation
plot(shapley_rf_ElBagre145)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre192 <- Shapley$new(predictor_shapley, x.interest = X[192, ]) # 192th observation
plot(shapley_rf_ElBagre192)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre216 <- Shapley$new(predictor_shapley, x.interest = X[216, ]) # 216th observation
plot(shapley_rf_ElBagre216)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_ElBagre249 <- Shapley$new(predictor_shapley, x.interest = X[249, ]) # 249th observation
plot(shapley_rf_ElBagre249)


#######
#######Quibdo
muni<- read.csv("Quibdo.csv", sep = ",")
munici_Quibdo <- round(muni, 2)
str(munici_Quibdo)

munici_Quibdo$Peak.of.cases <- as.factor(munici_Quibdo$Peak.of.cases)
levels(munici_Quibdo$Peak.of.cases) = c("No.Peak", "Peak")

summary(munici_Quibdo)

#Since the response variable is a binary categorical variable, you need to make sure the training data has approximately equal proportion of classes.
table(munici_Quibdo$Peak.of.cases)
'%ni%' <- Negate('%in%')  # define 'not in' func

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici_Quibdo$Peak.of.cases, p=0.7, list = F)  # 70% training data
trainData <- munici_Quibdo[trainDataIndex, ]
table(trainData$Peak.of.cases)
testData <- munici_Quibdo[-trainDataIndex, ]
table(testData$Peak.of.cases)


# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)


#  model with caret
set.seed(99)  # for reproducibility
xgb_trcontrol$sampling <- "smote"
xgboost_model = caret::train(Peak.of.cases ~ ., data = trainData, 
                             method = "xgbTree",
                             metric = "Sensitivity",
                             prob.model = TRUE, na.action = na.omit, 
                             trControl = xgb_trcontrol,
                             tuneGrid = xgbGrid)

# performance metrics
gbm.pred_Quibdo <- predict(xgboost_model, testData, na.action = na.pass)
ConfMat_Quibdo <- caret::confusionMatrix(gbm.pred_Quibdo, testData$Peak.of.cases, positive = "Peak")
print(ConfMat_Quibdo)


###Loop for ROC AUC of each fold
fold <- kfold(trainData, k=10)
auc<-rep(NA,10)
for (i in 1:10){
  training_fold <- trainData[fold == i, ]
  test_fold <- trainData[fold != i, ]
  y_pred <- predict(xgboost_model, newdata = test_fold, type="prob", na.action = na.pass)
  auc[i] <- result.roc <- roc(test_fold$Peak.of.cases, y_pred$Peak, direction = "<")$auc
}
aucXgboostk10_Quibdo <- auc
aucXgboostk10Mean_Quibdo <- mean(auc)


###### Interpretation
X <- munici_Quibdo%>%
  dplyr::select(- Peak.of.cases) %>%
  as.data.frame()

# Permutation Feature Importance
predictor_imp <- Predictor$new(xgboost_model, data = X, y = munici_Quibdo$Peak.of.cases, type = "prob")
importace_rf_Quibdo = FeatureImp$new(predictor_imp, loss = "ce", compare = 'difference', n.repetitions = 500)
importace_rf_Quibdo$plot()


# Plot feature interactions
predictor_inter <- Predictor$new(xgboost_model, data = X, type = "prob")
interactions_Quibdo <- Interaction$new(predictor_inter)
interactions_Quibdo$plot()

# Shapley values
predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo317 <- Shapley$new(predictor_shapley, x.interest = X[317, ]) # 317th observation
plot(shapley_rf_Quibdo317) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo322 <- Shapley$new(predictor_shapley, x.interest = X[322, ]) # 322th observation
plot(shapley_rf_Quibdo322) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo386 <- Shapley$new(predictor_shapley, x.interest = X[386, ]) # 386th observation
plot(shapley_rf_Quibdo386) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo403 <- Shapley$new(predictor_shapley, x.interest = X[403, ]) # 403th observation
plot(shapley_rf_Quibdo403)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo424 <- Shapley$new(predictor_shapley, x.interest = X[424, ]) # 424th observation
plot(shapley_rf_Quibdo424)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo447 <- Shapley$new(predictor_shapley, x.interest = X[447, ]) # 447th observation
plot(shapley_rf_Quibdo447)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo482 <- Shapley$new(predictor_shapley, x.interest = X[482, ]) # 482th observation
plot(shapley_rf_Quibdo482)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Quibdo494 <- Shapley$new(predictor_shapley, x.interest = X[494, ]) # 494th observation
plot(shapley_rf_Quibdo494)




#######
#######Tierraalta
muni<- read.csv("Tierraalta.csv", sep = ",")
munici_Tierraalta <- round(muni, 2)
str(munici_Tierraalta)

munici_Tierraalta$Peak.of.cases <- as.factor(munici_Tierraalta$Peak.of.cases)
levels(munici_Tierraalta$Peak.of.cases) = c("No.Peak", "Peak")

summary(munici_Tierraalta)

#Since the response variable is a binary categorical variable, you need to make sure the training data has approximately equal proportion of classes.
table(munici_Tierraalta$Peak.of.cases)
'%ni%' <- Negate('%in%')  # define 'not in' func

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici_Tierraalta$Peak.of.cases, p=0.7, list = F)  # 70% training data
trainData <- munici_Tierraalta[trainDataIndex, ]
table(trainData$Peak.of.cases)
testData <- munici_Tierraalta[-trainDataIndex, ]
table(testData$Peak.of.cases)


# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)


#  model with caret
set.seed(99)  # for reproducibility
xgb_trcontrol$sampling <- "smote"
xgboost_model = caret::train(Peak.of.cases ~ ., data = trainData, 
                             method = "xgbTree",
                             metric = "Sensitivity",
                             prob.model = TRUE, na.action = na.omit, 
                             trControl = xgb_trcontrol,
                             tuneGrid = xgbGrid)

# performance metrics
gbm.pred_Tierraalta <- predict(xgboost_model, testData, na.action = na.pass)
ConfMat_Tierraalta <- caret::confusionMatrix(gbm.pred_Tierraalta, testData$Peak.of.cases, positive = "Peak")
print(ConfMat_Tierraalta)

###Loop for ROC AUC of each fold
fold <- kfold(trainData, k=10)
auc<-rep(NA,10)
for (i in 1:10){
  training_fold <- trainData[fold == i, ]
  test_fold <- trainData[fold != i, ]
  y_pred <- predict(xgboost_model, newdata = test_fold, type="prob", na.action = na.pass)
  auc[i] <- result.roc <- roc(test_fold$Peak.of.cases, y_pred$Peak, direction = "<")$auc
}
aucXgboostk10_Tierraalta <- auc
aucXgboostk10Mean_Tierraalta <- mean(auc)


###### Interpretation
X <- munici_Tierraalta%>%
  dplyr::select(- Peak.of.cases) %>%
  as.data.frame()

# Permutation Feature Importance
predictor_imp <- Predictor$new(xgboost_model, data = X, y = munici_Tierraalta$Peak.of.cases, type = "prob")
importace_rf_Tierraalta = FeatureImp$new(predictor_imp, loss = "ce", compare = 'difference', n.repetitions = 500)
importace_rf_Tierraalta$plot()


# Plot feature interactions
predictor_inter <- Predictor$new(xgboost_model, data = X, type = "prob")
interactions_Tierraalta <- Interaction$new(predictor_inter)
interactions_Tierraalta$plot()

# Shapley values 
predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta83 <- Shapley$new(predictor_shapley, x.interest = X[83, ]) # 83th observation
plot(shapley_rf_Tierraalta83) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta89 <- Shapley$new(predictor_shapley, x.interest = X[89, ]) # 89th observation
plot(shapley_rf_Tierraalta89) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta121 <- Shapley$new(predictor_shapley, x.interest = X[121, ]) # 121th observation
plot(shapley_rf_Tierraalta121) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta136 <- Shapley$new(predictor_shapley, x.interest = X[136, ]) # 136th observation
plot(shapley_rf_Tierraalta136)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta536 <- Shapley$new(predictor_shapley, x.interest = X[536, ]) # 536th observation
plot(shapley_rf_Tierraalta536)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta553 <- Shapley$new(predictor_shapley, x.interest = X[553, ]) # 553th observation
plot(shapley_rf_Tierraalta553)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta567 <- Shapley$new(predictor_shapley, x.interest = X[567, ]) # 567th observation
plot(shapley_rf_Tierraalta567)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Tierraalta572 <- Shapley$new(predictor_shapley, x.interest = X[572, ]) # 572th observation
plot(shapley_rf_Tierraalta572)




#######
#######PtoLibertador
muni<- read.csv("PtoLibertador.csv", sep = ",")
munici_PtoLibertador <- round(muni, 2)
str(munici_PtoLibertador)

munici_PtoLibertador$Peak.of.cases <- as.factor(munici_PtoLibertador$Peak.of.cases)
levels(munici_PtoLibertador$Peak.of.cases) = c("No.Peak", "Peak")

summary(munici_PtoLibertador)

#Since the response variable is a binary categorical variable, you need to make sure the training data has approximately equal proportion of classes.
table(munici_PtoLibertador$Peak.of.cases)
'%ni%' <- Negate('%in%')  # define 'not in' func

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici_PtoLibertador$Peak.of.cases, p=0.7, list = F)  # 70% training data
trainData <- munici_PtoLibertador[trainDataIndex, ]
table(trainData$Peak.of.cases)
testData <- munici_PtoLibertador[-trainDataIndex, ]
table(testData$Peak.of.cases)


# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)


#  model with caret
set.seed(99)  # for reproducibility
xgb_trcontrol$sampling <- "smote"
xgboost_model = caret::train(Peak.of.cases ~ ., data = trainData, 
                             method = "xgbTree",
                             metric = "Sensitivity",
                             prob.model = TRUE, na.action = na.omit, 
                             trControl = xgb_trcontrol,
                             tuneGrid = xgbGrid)

# performance metrics
gbm.pred_PtoLibertador <- predict(xgboost_model, testData, na.action = na.pass)
ConfMat_PtoLibertador <- caret::confusionMatrix(gbm.pred_PtoLibertador, testData$Peak.of.cases, positive = "Peak")
print(ConfMat_PtoLibertador)


###Loop for ROC AUC of each fold
fold <- kfold(trainData, k=10)
auc<-rep(NA,10)
for (i in 1:10){
  training_fold <- trainData[fold == i, ]
  test_fold <- trainData[fold != i, ]
  y_pred <- predict(xgboost_model, newdata = test_fold, type="prob", na.action = na.pass)
  auc[i] <- result.roc <- roc(test_fold$Peak.of.cases, y_pred$Peak, direction = "<")$auc
}
aucXgboostk10_PtoLibertador <- auc
aucXgboostk10Mean_PtoLibertador <- mean(auc)


###### Interpretation
X <- munici_PtoLibertador%>%
  dplyr::select(- Peak.of.cases) %>%
  as.data.frame()

# Permutation Feature Importance
predictor_imp <- Predictor$new(xgboost_model, data = X, y = munici_PtoLibertador$Peak.of.cases, type = "prob")
importace_rf_PtoLibertador = FeatureImp$new(predictor_imp, loss = "ce", compare = 'difference', n.repetitions = 500)
importace_rf_PtoLibertador$plot()


# Plot feature interactions
predictor_inter <- Predictor$new(xgboost_model, data = X, type = "prob")
interactions_PtoLibertador <- Interaction$new(predictor_inter)
interactions_PtoLibertador$plot()

# Shapley values 
predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador3 <- Shapley$new(predictor_shapley, x.interest = X[3, ]) # 3th observation
plot(shapley_rf_PtoLibertador3) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador10 <- Shapley$new(predictor_shapley, x.interest = X[10, ]) # 10th observation
plot(shapley_rf_PtoLibertador10) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador18 <- Shapley$new(predictor_shapley, x.interest = X[18, ]) # 18th observation
plot(shapley_rf_PtoLibertador18) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador65 <- Shapley$new(predictor_shapley, x.interest = X[65, ]) # 65th observation
plot(shapley_rf_PtoLibertador65)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador94 <- Shapley$new(predictor_shapley, x.interest = X[94, ]) # 94th observation
plot(shapley_rf_PtoLibertador94)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador111 <- Shapley$new(predictor_shapley, x.interest = X[111, ]) # 111th observation
plot(shapley_rf_PtoLibertador111)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador135 <- Shapley$new(predictor_shapley, x.interest = X[135, ]) # 135th observation
plot(shapley_rf_PtoLibertador135)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_PtoLibertador540 <- Shapley$new(predictor_shapley, x.interest = X[540, ]) # 540th observation
plot(shapley_rf_PtoLibertador540)




#######
#######Caceres
muni<- read.csv("Caceres.csv", sep = ",")
munici_Caceres <- round(muni, 2)
str(munici_Caceres)

munici_Caceres$Peak.of.cases <- as.factor(munici_Caceres$Peak.of.cases)
levels(munici_Caceres$Peak.of.cases) = c("No.Peak", "Peak")

summary(munici_Caceres)

#Since the response variable is a binary categorical variable, you need to make sure the training data has approximately equal proportion of classes.
table(munici_Caceres$Peak.of.cases)
'%ni%' <- Negate('%in%')  # define 'not in' func

# Prep Training and Test data
set.seed(999)
trainDataIndex <- createDataPartition(munici_Caceres$Peak.of.cases, p=0.7, list = F)  # 70% training data
trainData <- munici_Caceres[trainDataIndex, ]
table(trainData$Peak.of.cases)
testData <- munici_Caceres[-trainDataIndex, ]
table(testData$Peak.of.cases)


# XGBoost model
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE
)

xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree=seq(0.5, 0.9, length.out=5),
                       #Values below are by default in sklearn-api
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)


#  model with caret
set.seed(99)  # for reproducibility
xgb_trcontrol$sampling <- "smote"
xgboost_model = caret::train(Peak.of.cases ~ ., data = trainData, 
                             method = "xgbTree",
                             metric = "Sensitivity",
                             prob.model = TRUE, na.action = na.omit, 
                             trControl = xgb_trcontrol,
                             tuneGrid = xgbGrid)

# performance metrics
gbm.pred_Caceres <- predict(xgboost_model, testData, na.action = na.pass)
ConfMat_Caceres <- caret::confusionMatrix(gbm.pred_Caceres, testData$Peak.of.cases, positive = "Peak")
print(ConfMat_Caceres)


###Loop for ROC AUC of each fold
fold <- kfold(trainData, k=10)
auc<-rep(NA,10)
for (i in 1:10){
  training_fold <- trainData[fold == i, ]
  test_fold <- trainData[fold != i, ]
  y_pred <- predict(xgboost_model, newdata = test_fold, type="prob", na.action = na.pass)
  auc[i] <- result.roc <- roc(test_fold$Peak.of.cases, y_pred$Peak, direction = "<")$auc
}
aucXgboostk10_Caceres <- auc
aucXgboostk10Mean_Caceres <- mean(auc)


###### Interpretation
X <- munici_Caceres%>%
  dplyr::select(- Peak.of.cases) %>%
  as.data.frame()

# Permutation Feature Importance
predictor_imp <- Predictor$new(xgboost_model, data = X, y = munici_Caceres$Peak.of.cases, type = "prob")
importace_rf_Caceres = FeatureImp$new(predictor_imp, loss = "ce", compare = 'difference', n.repetitions = 500)
importace_rf_Caceres$plot()


# Plot feature interactions
predictor_inter <- Predictor$new(xgboost_model, data = X, type = "prob")
interactions_Caceres <- Interaction$new(predictor_inter)
interactions_Caceres$plot()

# Shapley values 
predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres20 <- Shapley$new(predictor_shapley, x.interest = X[20, ]) # 20th observation
plot(shapley_rf_Caceres20) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres65 <- Shapley$new(predictor_shapley, x.interest = X[65, ]) # 65th observation
plot(shapley_rf_Caceres65) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres79 <- Shapley$new(predictor_shapley, x.interest = X[79, ]) # 79th observation
plot(shapley_rf_Caceres79) 

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres94 <- Shapley$new(predictor_shapley, x.interest = X[94, ]) # 94th observation
plot(shapley_rf_Caceres94)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres113 <- Shapley$new(predictor_shapley, x.interest = X[113, ]) # 113th observation
plot(shapley_rf_Caceres113)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres132 <- Shapley$new(predictor_shapley, x.interest = X[132, ]) # 132th observation
plot(shapley_rf_Caceres132)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres275 <- Shapley$new(predictor_shapley, x.interest = X[275, ]) # 275th observation
plot(shapley_rf_Caceres275)

predictor_shapley <- Predictor$new(xgboost_model, data = X, type = "prob", class = 2)#class 2 = "Peak"
shapley_rf_Caceres289 <- Shapley$new(predictor_shapley, x.interest = X[289, ]) # 289th observation
plot(shapley_rf_Caceres289)

