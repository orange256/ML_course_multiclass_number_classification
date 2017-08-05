# -- Machine Learning 2017 -- 
# [HW 4] : Supporting Vector machine (for classification)
# [Deadline] : 2017/5/2
# [Model] : C-SVM with linear model

#==================================================================================
# Functions 
#==================================================================================

# This function can help you to compute error rate
# data : output dataframe
# n    : n-th column
get_error_rate <- function(data,n){
  
  hit_table <- table(data[,c(1,n)])
  
  hit_rate <- sum(diag(hit_table)) / nrow(data)
  
  error_rate <- 1 - hit_rate
  
  print(paste0("Error rate of ",colnames(data)[n]," : ",error_rate))
}
#====================================================================================
# Main 
#====================================================================================

# [0] path & packages ---------------------------------------------------------------
  path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/HW4/") # Windows
  path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/HW4/") # Mac
  
  library(magrittr)
  library(dplyr)
  library(ggplot2)
  library(e1071) # libSVM

# [1] get row data --------------------------------------------------------------------
  X_train <- read.csv(paste0(path,"data/X_train.csv"),header = F)
  T_train <- read.csv(paste0(path,"data/T_train.csv"),header = F)
  
  X_test <- read.csv(paste0(path,"data/X_test.csv"),header = F)
  T_test <- read.csv(paste0(path,"data/T_test.csv"),header = F)

  # cbine X_train & T_train to fulldata 5000x(1+784)
  colnames(T_train) <- "V0"
  T_train$V0 <- as.factor(T_train$V0)
  fulldata <- cbind.data.frame(T_train,X_train)

  colnames(T_test) <- "V0"
  T_test$V0 <- as.factor(T_test$V0)

# [2] PCA  ---------------------------------------------------------------------------
  # get PCA parameters from X_train
  PCA <- prcomp(X_train)
  
  # Do PCA rotation, and get X_train_PCA
  X_train_PCA <- as.data.frame(PCA$x)
  fulldata_PCA <- cbind.data.frame(T_train,X_train_PCA)
  
# [3] C-SVM with linear model -------------------------------------------------
  C_SVM_model_linear <- svm(V0 ~ .,data = fulldata_PCA, type = "C-classification", kernel = "linear")

# [4] testing data ---------------------------------------------------------------
  C_linear_output <- predict(C_SVM_model_linear,X_test)

  # get error rate
  get_error_rate(cbind(T_test, C_linear_output),2)

# [5] Plots ----------------------------------------------------------------------------

  # get SV labels from model
  SV_label <- attributes(C_SVM_model_linear$SV)$dimnames[[1]]
  SV_df <- data.frame(V0 = T_train[c(SV_label),1]) %>% cbind.data.frame(.,C_SVM_model_linear$SV)
  
  # fulldata
  plot(C_SVM_model_linear, data = fulldata_PCA, formula =  PC1 ~ PC2)
  
  # SV
  plot(C_SVM_model_linear, data = SV_df, formula =  PC1 ~ PC2)
  
  # outliers
  test_output <- cbind(T_test, C_linear_output) %>% as.data.frame()
  test_output <- test_output %>% mutate(hit = V0==C_linear_output)
  
  X_test_PCA <- predict(PCA, X_test) %>% as.data.frame()
  fulltestdata_PCA <- cbind.data.frame(test_output, X_test_PCA)
  outliers <- fulltestdata_PCA %>% filter(hit==F)
  

  plot(C_SVM_model_linear, data = outliers[,-c(2,3)], formula =  PC1 ~ PC2)
  
  
  ggplot(fulldata_PCA,aes(x=PC2, y=PC1, group = V0) ) + 
    geom_point(aes(color = V0, shape = V0))+
    stat_ellipse(aes(color = V0, size=2))
  

# [6] other discussion (PCA) ------------------------------------------------
  # compute variance and it's %
  vars <- (PCA$sdev)^2  
  props <- vars / sum(vars)    
  
  
  # plot cumulative props
  cumulative.props <- cumsum(props)  
  plot(cumulative.props)

  