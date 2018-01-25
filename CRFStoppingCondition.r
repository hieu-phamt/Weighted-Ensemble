##############################################################
#                                                            #
#                   Cesaro Random Forest                     #
#                                                            #
#                                                            #
# LAST UPDATE: 1/17/18  PHAM                                 #
##############################################################

library(dplyr)
library(optimbase)
library(ranger)
library(foreach)
library(doParallel)


#ModifiedRF will prints matrix of first 10 votes and returns full matrix
modifiedRF = function(data, class.response, numtree = 10000, start.criteria = 100, next.trees = 10){
  
  # Rename classes
  data[,which(colnames(data) == class.response)] <- as.numeric(data[,which(colnames(data) == class.response)])
  data[,which(colnames(data) == class.response)] <- as.factor(data[,which(colnames(data) == class.response)])
  
  # Creating training and testing sets globally
  rand.sample <- sample(nrow(data),floor(3/4*nrow(data)))
  trainData <- data[rand.sample,]
  testData <- data[-rand.sample,]
  
  # Formula
  f <- paste(names(data)[which(colnames(data) == class.response)], "~.", collapse = "")
  myFormula <- as.formula(f)
  
  #Random Forest 1 tree at a time
  collection <- function(myFormul = myFormula){
    rf <- ranger(myFormul, data = trainData, num.trees = 1) # Random forest one tree at a time
    rf.results <- predict(rf,testData) # Prediction for that one tree
    results.rf <- c(rf$prediction.error,rf.results$predictions)
    return(results.rf)
  }
  
  results <- foreach(i = 1:numtree, .combine= 'rbind', .packages = "ranger") %dopar% {
    collection(myFormul = myFormula)
  }
  
  df.rf <- data.frame(results)
  treeName <- numeric(numtree)
  for(k in 1:numtree){treeName[k] = paste0("T",k)} # Function to name rows
  rownames(df.rf) <- treeName # Rename Rows
  colnames(df.rf) <- c("OOB",rownames(testData)) # Rename columns
  df.rf.trees <- data.frame(df.rf$OOB,1:numtree)
  
  ############################# 
  #                           #
  # Construct Data Frame      #     
  #                           #  
  ############################# 

  ############################# 
  #                           #
  #   Stopping Condition      #     
  #                           #  
  ############################# 
  
  k <- start.criteria
  while(k <= numtree){
    if(k == numtree){
      print(paste("Stopping Criteria Not Met After", k," Trees"))
    }
    
    temp.df <- df.rf.trees[1:k,]
    temp.df <- temp.df[order(temp.df$df.rf.OOB),]
    next.obs <- df.rf.trees[(k+1):(k+next.trees),]
    new.temp.df <- rbind(temp.df,next.obs)
    new.temp.df <- new.temp.df[order(new.temp.df$df.rf.OOB),]
    twenty.percent <- floor(.2*nrow(new.temp.df))
    twenty.vec <- new.temp.df$X1.numtree[1:twenty.percent]

    if(length(intersect(c((k+1):(k+next.trees)),twenty.vec)) == 0 ){
      df.rf.oob <- df.rf[1:(k+next.trees),]
      df.rf.oob <- df.rf.oob[order(df.rf.oob$OOB),]
      break
    }
    k <- k + 10
  }
  
  
  ################################
  # Only Data Frames That Matter #
  ################################
  
  # OOB data frame with only votes
  df.rf.oob.votes <- subset(df.rf.oob,select = -OOB)
  df.rf.votes <- subset(df.rf, select = -OOB)
  df.rf.votes <- df.rf.votes[1:nrow(df.rf.oob.votes),]
  
  ############################## 
  #                            #
  # Calculated Weighted Votes  #     
  #                            #  
  ############################## 
  
  # Weights function
  weights <- function(n){
    weight <- numeric(n)
    harmonic <- sapply(1:n,FUN = function(x) 1/x)
    for(j in 1:n){
      weight[j] <- sum(harmonic[j:n])
    }
    return(weight)
  }
  
  # Weights based on \sum_{1=j}^{n} \dfrac{1}{n}
  weight <- weights(nrow(df.rf.oob.votes))
  ##############################
  #       Random Forest        #
  ##############################
  
  vote.rf.weight <- apply(df.rf.oob.votes,2,FUN = function(x) sum(weight*x)/sum(weight))
  vote.rf.no.weight <- apply(df.rf.oob.votes,2,mean)
  
  vote.rf.weight <- data.frame(vote.rf.weight)
  vote.rf.no.weight <- data.frame(vote.rf.no.weight)
  
  #######################################
  #       Random Forest Accuracy        #
  #######################################
  
  rf.vote.weight <- vote.rf.weight %>% mutate(class = ifelse(vote.rf.weight <= 1.5,levels(data[,which(colnames(data) == class.response)])[1],levels(data[,which(colnames(data) == class.response)])[2]))
  rf.vote.no.weight <- vote.rf.no.weight %>% mutate(class = ifelse(vote.rf.no.weight <= 1.5,levels(data[,which(colnames(data) == class.response)])[1],levels(data[,which(colnames(data) == class.response)])[2]))
  
  ############################## 
  #                            #
  #      Compute Accuracy      #     
  #                            #  
  ############################## 
  
  acc.rf.weight <- sum(rf.vote.weight$class == testData[,which(colnames(data) == class.response)]) / nrow(rf.vote.weight)
  acc.rf.no.weight <- sum(rf.vote.no.weight$class == testData[,which(colnames(data) == class.response)]) / nrow(rf.vote.no.weight)
  
  output <- transpose(c(acc.rf.weight,acc.rf.no.weight,k))
  df.out <- data.frame(rownames(df.rf.oob),df.rf.oob[,1])
  colnames(df.out) <- c("Tree","OOB")
  return(list(df.out,output))
}


#########################Sonar################
dataSonar = read.csv("sonar.csv")

cl <- makeCluster(detectCores())
registerDoParallel(cl)

df.tree <- NA
df.results <- NA
for(i in 1:100){
  t = modifiedRF(dataSonar,"ending",numtree = 10000, start.criteria = 100, next.trees = 25)
  #df.tree <- cbind(df.tree,t[[1]])
  df.results <- cbind(df.results,t[[2]])
}

df.results <- data.frame(df.results[,2:ncol(df.results)])
df.results$STD <- apply(df.results,1,sd)
df.results$Mean <- rowMeans(df.results[,1:(ncol(df.results)-1)])
min(df.results[3,1:100])
max(df.results[3,1:100])
median(as.numeric(df.results[3,1:100]))

write.csv(df.results,"Results Sonar 100.csv")
write.csv(df.tree,"Tree Sonar 100.csv")

stopCluster(cl)
