##########################Classifying Handwritten digits##########################################

#To start of I have already given name to the columns for dataset in excel itself as the no of columns are huge
#First column Being Label, followed by Pixel_1 to Pixel_784
#This has been done for Train as well as Test dataset

#Loading Train and Test data
train <- read.csv("mnist_train.csv", stringsAsFactors = F)
test <- read.csv("mnist_test.csv", stringsAsFactors = F)

#Loading necessary libraries

#install.packages('kernlab')
library(kernlab)
#install.packages('readr')
library(readr)
#install.packages('caret')
#install.packages('lattice')
library(lattice)
#install.packages('ggplot2')
library(ggplot2)
library(caret)
#install.packages('dplyr')
library(dplyr)
#install.packages('gridExtra')
library(gridExtra)
#install.packages('devtools')
library(devtools)
install.packages('e1071')
library(e1071)


#structure of the data
str(train)

#printing first few records
head(train)

#checking missing value
sapply(train, function(x) sum(is.na(x))) #there are no missing values


#check for duplicate data
sum(duplicated(1:nrow(train))) #there are no such records

#making label as factor as its the column for which we are going to predict
train$Label <- as.factor(train$Label)
test$Label <- as.factor(test$Label)

#Now as the data is huge lets create sample out of it and perform all the operation on that
set.seed(100)

df2 <- lapply(split(train, train$Label),function(subdf) subdf[sample(1:nrow(subdf), 600),])
train_sample <- do.call('rbind', df2)

#train_index <- sample(1:nrow(train),4000)
#train_sample <- train[train_index,]

#scaling the data
train_sample[,2:ncol(train_sample)]<- apply(train_sample[,2:ncol(train_sample)], 2, function(y) (y - mean(y)) / sd(y) ^ as.logical(sd(y)))

#constructing model

#using linear kernel
model_linear <- ksvm(Label~ ., data=train_sample, scale = FALSE, kernel = "vanilladot")
eval_linear <- predict(model_linear,test)

#confusion matrix - Linear Kernel
confusionMatrix(eval_linear,test$Label)
#Accuracy : 0.8287

#using polynomial kernel
model_polynomial <- ksvm(Label~ ., data=train_sample, scale = FALSE, kernel = "polydot")
eval_polynomial <- predict(model_polynomial,test)


#confusion matrix - Linear Kernel
confusionMatrix(eval_polynomial,test$Label)
#Accuracy : 0.8287

#using RBF kernel
model_rbf <- ksvm(Label~ ., data=train_sample, scale = FALSE, kernel = "rbfdot")
eval_rbf <- predict(model_rbf,test)


#confusion matrix - Linear Kernel
confusionMatrix(eval_rbf,test$Label)
#Accuracy : 0.1028

#so from the above we get to know that either linear or polynomial model is good to proceed


#########Hyperparameter Tunning and cross validation for Linear Model#########

trainControl <- trainControl(method="cv", number=5)

metric <- "Accuracy"

set.seed(100)

# making a grid of C values. 
grid <- expand.grid(C=seq(0.01, 0.10, by=0.01))

# Performing 5-fold cross validation
fit.svm_linear <- train(Label~., data=train_sample, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm_linear)

# Plotting model results
plot(fit.svm_linear)
#Best Tune at 0.01
#Accuracy : 0.92

################

# Valdiating the model after cross validation on test data

evaluate_linear_test<- predict(fit.svm_linear, test)
confusionMatrix(evaluate_linear_test, test$Label)
#Accuracy : 0.833


#########Hyperparameter Tunning and cross validation for Polynomial Model#########

trainControl_poly <- trainControl(method="cv", number=5)

metric_poly <- "Accuracy"

set.seed(100)

# making a grid of C values. 
grid_poly <- expand.grid(degree=2,scale=0.2,C=seq(0.001, 0.005, by=0.001))

# Performing 5-fold cross validation
fit.svm_polynomial <- train(Label~., data=train_sample, method="svmPoly", metric=metric_poly, 
                        tuneGrid=grid_poly, trControl=trainControl_poly)

# Printing cross validation result
print(fit.svm_polynomial)

# Plotting model results
plot(fit.svm_polynomial)


################

# Valdiating the model after cross validation on test data

evaluate_polynomial_test<- predict(fit.svm_polynomial, test)
confusionMatrix(evaluate_polynomial_test, test$Label)
#Accuracy : 0.2134



########################################Result#####################################################
#So we have tested Linear as well as Polynomial model and according to the test results we can 
#clearly see that Linear Model is best fitted for classifying handwritten digits

#so finalising on Linear model with
##Accuracy : 0.92 & Best Tune at 0.01
#for test data Accuracy : 0.833
