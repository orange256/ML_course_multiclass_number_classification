# -- Machine Learning 2017 -- 
# [HW 5] : Random Forest (for classification)
# [Deadline] : 2017/5/16

#====================================================================================
# Main 
#====================================================================================

# [0] path & packages ---------------------------------------------------------------
  path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/HW5/") # Windows
  path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/HW5/") # Mac
  
  library(magrittr)
  library(dplyr)
  library(ggplot2)
  library(randomForest)

# [1] get row data -----------------------------------------------------------------
  X_train <- read.csv(paste0(path,"data/X_train.csv"),header = F)
  T_train <- read.csv(paste0(path,"data/T_train.csv"),header = F)
  
  X_test <- read.csv(paste0(path,"data/X_test.csv"),header = F)
  T_test <- read.csv(paste0(path,"data/T_test.csv"),header = F)
  
  colnames(T_train) <- "V0"
  T_train$V0 <- as.factor(T_train$V0)
  fulldata <- cbind.data.frame(T_train,X_train)
  

# [2] Feature Extractor (PCA)----------------------------------------------------
# get PCA parameters from X_train
  PCA <- prcomp(X_train)
  X_train_PCA <- as.data.frame(PCA$x)
  fulldata_PCA <- cbind.data.frame(T_train,X_train_PCA)

# [3] Random Forest model -------------------------------------------------
  set.seed(12345)
  
  fit <- randomForest(V0 ~ PC1 + PC2 + PC3 + PC4 + PC5,
                      data = fulldata_PCA,
                      importance = TRUE,
                      ntree = 100,
                      nodesize = 1000)
  fit
# [4] testing data ---------------------------------------------------------------
  X_test_PCA <- predict(PCA, X_test) %>% as.matrix()
  Prediction <- predict(fit, X_test_PCA)
  model_output <- cbind(T_test, Prediction)
  
  
  hit_table <- table(model_output)
  hit_table
  
  hit_rate <- sum(diag(hit_table)) / nrow(model_output)
  
  error_rate <- 1 - hit_rate
  error_rate
  
# [5] Plots ----------------------------------------------------------------------------
  # error curve (Number of trees)
  plot(fit, log="y")

  # add legend
  layout(matrix(c(1,2),nrow=1), width=c(7,2)) 
  par(mar=c(5,4,4,0)) #No margin on the right side
  plot(fit, log="y")
  par(mar=c(5,0,4,2)) #No margin on the left side
  plot(c(0,1),type="n", axes=F, xlab="", ylab="")
  legend("top", colnames(fit$err.rate),col=1:6,cex=0.8,fill=1:6)
  