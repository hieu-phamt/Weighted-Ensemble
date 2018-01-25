##############################################################
#                                                            #
#              Weighted Ensemble: CART, KNN, NB              #
#                                                            #
# LAST UPDATE: 10-9-17                                       #
##############################################################

library(caret)
library(klaR)
library(optimbase)
library(rpart)
library(dplyr)
library(ranger)
library(foreach)
library(doParallel)

# Parallel
#cl <- makeCluster(detectCores())
#registerDoParallel(cl)

# Uncomment line below in case of emergency
#stopCluster(cl)


# data: data set
# class: response variable as string
# size: number of models in ensemble
# nfold: number of folds in cross validation
# p: tuning parameter
# num.cores: number of cores to use

ensemble <- function(data,class,size = 500, nfolds = 3, p = 1, num.cores = detectCores()){
  
  # Rename classes
  data[,which(colnames(data) == class)] <- as.numeric(data[,which(colnames(data) == class)])
  data[,which(colnames(data) == class)] <- as.factor(data[,which(colnames(data) == class)])
  
  # Creating training and testing sets globally
  rand.sample <- sample(nrow(data),floor(3/4*nrow(data)))
  trainData <- data[rand.sample,]
  testData <- data[-rand.sample,]
  
  # Formula
  f <- paste(names(data)[which(colnames(data) == class)], "~.", collapse = "")
  myFormula <- as.formula(f)
  
  
  collection <- function(myFormul = myFormula){
    bag.sample <- sample(nrow(trainData),nrow(trainData),replace = T) # Bagging sample numbers for training data
    train.data.bag <- trainData[bag.sample,] # Bagged training data
    test.data.bag <- trainData[-bag.sample,] # Bagged testing data
    
    # CART
    cart.model <- train(myFormul, data = train.data.bag,method = "rpart") # rpart on training data
    train_control <- trainControl(method="cv", number = nfolds) # Cross validation
    model <- train(myFormul, data=train.data.bag, trControl=train_control, method="rpart")
    
    # Naive Bayes
    nb.model <- train(myFormul,train.data.bag, method = "nb") # rpart on training data
    train_control <- trainControl(method="cv", number = nfolds) # Cross validation
    model <- train(myFormul, data=train.data.bag, trControl=train_control, method="nb")
    
    # K-Nearest Neighbors
    knn.model <- train(myFormul,data = train.data.bag, method = "knn",preProc = c("center","scale"),tuneLength = 10) #tuneLength = 10
    train_control <- trainControl(method="cv", number = nfolds) # Cross validation
    model <- train(myFormul, data=train.data.bag,preProc = c("center","scale"), trControl=train_control, method="knn")
    
    # Random Forest
    rf <- ranger(myFormul, data = train.data.bag, num.threads = num.cores, num.trees = 1) # Random forest one tree at a time
    rf.results <- predict(rf,testData, num.threads = num.cores) # Prediction for that one tree
    
    cart.results <- c(
      # Predict on test.data.bag to get OOB
      1 - sum(predict(cart.model, test.data.bag) == test.data.bag[,which(colnames(test.data.bag) == class)])/length(predict(cart.model, test.data.bag)), # OOB error rpart
      1 - mean(model$resample[,1]), # CV error
      predict(cart.model,testData) # Predict on actual test data
    )
    
     nb.results <- c(
      # Predict on test.data.bag to get OOB
      1 - sum(predict(nb.model, test.data.bag) == test.data.bag[,which(colnames(test.data.bag) == class)])/length(predict(nb.model, test.data.bag)), # OOB error rpart
      1 - mean(model$resample[,1]),  # CV error
      predict(nb.model,testData) # Predict on actual test data
    )
       
    knn.results <- c(
      # Predict on test.data.bag to get OOB
      1 - sum(predict(knn.model, test.data.bag) == test.data.bag[,which(colnames(test.data.bag) == class)])/length(predict(knn.model, test.data.bag)), # OOB error rpart
      1 - mean(model$resample[,1]),  # CV error
      predict(knn.model,testData) # Predict on actual test data
    )
    
    return(list(cart.results,nb.results,knn.results,rf.results$predictions))
  }
  
  # Combine function for the foreach loop
  comb <- function(x, ...) {
    lapply(seq_along(x),
           function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
  }  
  
results <- foreach(i = 1:size, .combine='comb', .multicombine=TRUE,.packages = c("caret","ranger"),
                   .init=list(list(), list(),list(),list())) %dopar% {
                     collection(myFormul = myFormula)
                     }

df.cart <- data.frame(do.call(rbind,results[[1]]))
df.nb <- data.frame(do.call(rbind,results[[2]]))
df.knn <- data.frame(do.call(rbind,results[[3]]))
df.rf <- data.frame(do.call(rbind,results[[4]]))
colnames(df.cart) <- c("OOB","CV",rownames(testData)) # Rename columns
colnames(df.nb) <- c("OOB","CV",rownames(testData)) # Rename columns
colnames(df.knn) <- c("OOB","CV",rownames(testData)) # Rename columns
colnames(df.rf) <- rownames(testData) # Rename columns
  
############################# 
#                           #
# Construct Data Frames     #     
#                           #  
############################# 

# Subset data frames for only OOB
df.cart.oob <- subset(df.cart, select = -CV)
df.nb.oob <- subset(df.nb, select = -CV)
df.knn.oob <- subset(df.knn, select = -CV)

# Subset data frames for only CV
df.cart.cv <- subset(df.cart, select = -OOB)
df.nb.cv <- subset(df.nb, select = -OOB)
df.knn.cv <- subset(df.knn, select = -OOB)

# Sort OOB by increasing
df.cart.oob <- df.cart.oob[order(df.cart.oob$OOB),]
df.nb.oob <- df.nb.oob[order(df.nb.oob$OOB),]
df.knn.oob <- df.knn.oob[order(df.knn.oob$OOB),]

# Sort CV by increasing
df.cart.cv <- df.cart.cv[order(df.cart.cv$CV),]
df.nb.cv <- df.nb.cv[order(df.nb.cv$CV),]
df.knn.cv <- df.knn.cv[order(df.knn.cv$CV),]

################################
# Only Data Frames That Matter #
################################

# OOB data frame with only votes
df.cart.oob <- subset(df.cart.oob,select = -OOB)
df.nb.oob <- subset(df.nb.oob,select = -OOB)
df.knn.oob <- subset(df.knn.oob,select = -OOB)

# CV data frame with only votes
df.cart.cv <- subset(df.cart.cv,select = -CV)
df.nb.cv <- subset(df.nb.cv,select = -CV)
df.knn.cv <- subset(df.knn.cv,select = -CV)

# data frame with no ordering only votes
df.cart.none <- subset(df.cart, select = -c(CV,OOB))
df.nb.none <- subset(df.nb, select = -c(CV,OOB))
df.knn.none <- subset(df.knn, select = -c(CV,OOB))

############################## 
#                            #
# Calculated Weighted Votes  #     
#                            #  
############################## 

# Weights function
weights <- function(n,p){
  weight <- numeric(n)
  harmonic <- sapply(1:n,FUN = function(x) 1/x^p)
  for(j in 1:n){
    weight[j] <- sum(harmonic[j:n])
  }
  return(weight)
}

# Weights based on \sum_{i=j}^{n} \dfrac{1}{n^p}
weight <- weights(size,p)

##############################
#            CART            #
##############################

vote.cart.oob <- apply(df.cart.oob,2,FUN = function(x) sum(weight*x)/sum(weight)) 

vote.cart.cv <- apply(df.cart.cv,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.cart.none <- apply(df.cart.none,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.cart.no.weight <- apply(df.cart.none,2,mean)

##############################
#        Naive Bayes         #
##############################

vote.nb.oob <- apply(df.nb.oob,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.nb.cv <- apply(df.nb.cv,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.nb.none <- apply(df.nb.none,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.nb.no.weight <- apply(df.nb.none,2,mean)

##############################
#             K-NN           #
##############################
vote.knn.oob <- apply(df.knn.oob,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.knn.cv <- apply(df.knn.cv,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.knn.none <- apply(df.knn.none,2,FUN = function(x) sum(weight*x)/sum(weight))

vote.knn.no.weight <- apply(df.knn.none,2,mean)
##############################
#       Random Forest        #
##############################
vote.rf <- apply(df.rf,2,mean)

############################## 
#                            #
#       Aggregate Votes      #     
#                            #  
############################## 

# Need to make votes into data frame to use dplyr
vote.cart.oob <- data.frame(vote.cart.oob)
vote.cart.cv <- data.frame(vote.cart.cv)
vote.cart.none <- data.frame(vote.cart.none)
vote.cart.no.weight <- data.frame(vote.cart.no.weight)
vote.nb.oob <- data.frame(vote.nb.oob)
vote.nb.cv <- data.frame(vote.nb.cv)
vote.nb.none <- data.frame(vote.nb.none)
vote.nb.no.weight <- data.frame(vote.nb.no.weight)
vote.knn.oob <- data.frame(vote.knn.oob)
vote.knn.cv <- data.frame(vote.knn.cv)
vote.knn.none <- data.frame(vote.knn.none)
vote.knn.no.weight <- data.frame(vote.knn.no.weight)
vote.rf <- data.frame(vote.rf)
##############################
#            CART            #
##############################

# Create new column based on condition of being <= 1.5. If <= 1.5 class 1 otherwise class 2.
cart.oob <- vote.cart.oob %>% mutate(class = ifelse(vote.cart.oob <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

cart.cv <- vote.cart.oob %>% mutate(class = ifelse(vote.cart.cv <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

cart.none <- vote.cart.oob %>% mutate(class = ifelse(vote.cart.none <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

cart.no.weight <- vote.cart.no.weight %>% mutate(class = ifelse(vote.cart.no.weight <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))
##############################
#         Naive Bayes        #
##############################
# Create new column based on condition of being <= 1.5. If <= 1.5 class 1 otherwise class 2.
nb.oob <- vote.nb.oob %>% mutate(class = ifelse(vote.nb.oob <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

nb.cv <- vote.nb.oob %>% mutate(class = ifelse(vote.nb.cv <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

nb.none <- vote.nb.oob %>% mutate(class = ifelse(vote.nb.none <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

nb.no.weight <- vote.nb.no.weight %>% mutate(class = ifelse(vote.nb.no.weight <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))
##############################
#            K-NN            #
##############################
# Create new column based on condition of being <= 1.5. If <= 1.5 class 1 otherwise class 2.
knn.oob <- vote.knn.oob %>% mutate(class = ifelse(vote.knn.oob <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

knn.cv <- vote.knn.oob %>% mutate(class = ifelse(vote.knn.cv <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

knn.none <- vote.knn.oob %>% mutate(class = ifelse(vote.knn.none <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))

knn.no.weight <- vote.knn.no.weight %>% mutate(class = ifelse(vote.knn.no.weight <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))
##############################
#       Random Forest        #
##############################
rf.vote <- vote.rf %>% mutate(class = ifelse(vote.rf <= 1.5,levels(data[,which(colnames(data) == class)])[1],levels(data[,which(colnames(data) == class)])[2]))
##############################
############################## 
#                            #
#      Compute Accuracy      #     
#                            #  
############################## 

##############################
#            CART            #
##############################
# Compute accuracy
acc.cart.oob <- sum(cart.oob$class == testData[,which(colnames(data) == class)] ) / nrow(cart.oob)
acc.cart.cv <- sum(cart.cv$class == testData[,which(colnames(data) == class)] ) / nrow(cart.cv)
acc.cart.none <- sum(cart.none$class == testData[,which(colnames(data) == class)] ) / nrow(cart.none)
acc.cart.no.weight <- sum(cart.no.weight$class == testData[,which(colnames(data) == class)] ) / nrow(cart.none)
##############################
#         Naive Bayes        #
##############################
# Compute accuracy
acc.nb.oob <- sum(nb.oob$class == testData[,which(colnames(data) == class)] ) / nrow(nb.oob)
acc.nb.cv <- sum(nb.cv$class == testData[,which(colnames(data) == class)] ) / nrow(nb.cv)
acc.nb.none <- sum(nb.none$class == testData[,which(colnames(data) == class)] ) / nrow(nb.none)
acc.nb.no.weight <- sum(cart.no.weight$class == testData[,which(colnames(data) == class)] ) / nrow(cart.none)
##############################
#            K-NN            #
##############################
# Compute accuracy
acc.knn.oob <- sum(knn.oob$class == testData[,which(colnames(data) == class)] ) / nrow(knn.oob)
acc.knn.cv <- sum(knn.cv$class == testData[,which(colnames(data) == class)] ) / nrow(knn.cv)
acc.knn.none <- sum(knn.none$class == testData[,which(colnames(data) == class)] ) / nrow(knn.none)
acc.knn.no.weight <- sum(cart.no.weight$class == testData[,which(colnames(data) == class)] ) / nrow(cart.none)
##############################
#       Random Forest        #
##############################
acc.rf <- sum(rf.vote$class == testData[,which(colnames(data) == class)]) / nrow(rf.vote)
# Compile results in this order: OOB, Cross-Validation, None
#CART, Naive Bayes, K-NN, single random forest
results <- transpose(c(acc.cart.oob,acc.cart.cv,acc.cart.none,acc.cart.no.weight,
                       acc.nb.oob,acc.nb.cv,acc.nb.none,acc.nb.no.weight,
                       acc.knn.oob,acc.knn.cv,acc.knn.none,acc.knn.no.weight,
                       acc.rf))

rm(acc.cart.oob,acc.cart.cv,acc.cart.none,acc.cart.no.weight,
  acc.nb.oob,acc.nb.cv,acc.nb.none,acc.nb.no.weight,
  acc.knn.oob,acc.knn.cv,acc.knn.none,acc.knn.no.weight,
  acc.rf,pred.rf,rf,
  cart.oob,cart.cv,cart.none,cart.no.weight,
  nb.oob,nb.cv,nb.none,nb.no.weight,
  knn.oob,knn.cv,knn.none,knn.no.weight,
  df.cart,df.cart.cv,df.cart.none,df.cart.oob,
  df.nb,df.nb.cv,df.nb.none,df.nb.oob,
  df.knn,df.knn.cv,df.knn.none,df.knn.oob,
  vote.cart.oob,vote.cart.cv,vote.cart.none,vote.cart.no.weight,
  vote.nb.oob,vote.nb.cv,vote.nb.none,vote.nb.no.weight,
  vote.knn.oob,vote.knn.cv,vote.knn.none,vote.knn.no.weight,
  testData,trainData,data,
  test.data.bag,train.data.bag,df.rf)

return(results)
}


