# Prediction Assignment Writeup
## Synopsis: 
This analysis examines the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  This project's goal is to use the data from the sensors to train a model that can predict how well the activity was performed.

### Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
  
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  
  
The data for this project come from this source:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.  
  

## Load Libraries



## Loading and Preprocessing the Data


```r
knitr::opts_chunk$set(echo = TRUE, results = "asis")
knit_hooks$set(webgl = hook_webgl)

## Center Justify All Plot Titles for ggplot
theme_update(plot.title = element_text(hjust = 0.5))

## Read activity.csv and format as tbl_df
actyTrainData <- read.csv("pml-training.csv", header = TRUE)
actyTrainData <- tbl_df(actyTrainData)

## Make a copy to use to partition
traindata <- tbl_df(actyTrainData)

## Read Column names
keepcolnames <-colnames(traindata)

## Create vector of column names to keep (remove summary statistic column names)
names_vector <- !(grepl("^(kurtosis)",keepcolnames) | grepl("^(skewness)",keepcolnames) | grepl("^(max)",keepcolnames) | grepl("^(min)",keepcolnames) | grepl("^(amplitude)",keepcolnames)| grepl("^(var)",keepcolnames) | grepl("^(avg)",keepcolnames) | grepl("^(stddev)",keepcolnames) )

## Select only column names with senor data
filtertraindata <- select(traindata,keepcolnames[names_vector])

## Remove first seven (7) columns of non-sensor data
filtertraindata <- select(filtertraindata,-c(1:7))

## Partition the Traing data into two Training sets for cross-validation
inTrain = createDataPartition(y=filtertraindata$classe, p = 0.7,list=FALSE)
training1 = filtertraindata[inTrain,]
training2 = filtertraindata[-inTrain,]
```


## Train the Model as Random Forest

```r
## Parallel Cluster Code from Class Forum/Discusson on methods using Parallel Processing:
## Class Forum notes references the following github repo:
## https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md

## Also, had to use: devtools::install_github('topepo/caret/pkg/caret') to update caret

## Set up Parallel Cluster  -- Allow one (1) core for MAC-OS:
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

## Set Training Control Parameters:
fitControl <- trainControl(method = "oob",number = 10, allowParallel = TRUE)

## Fit Classe to all Sensor data:
fit2 <- train(classe ~. , method="rf",data=training1, trControl = fitControl, importance = TRUE)

## Stop parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```

## Perform Model Review and Cross Validation:


```r
## Model Review
## Compare 1st Training Set with Model Predictions
conMatrix <- confusionMatrix(training1$classe, fit2$finalModel$predicted)
print(conMatrix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3899    4    2    0    1
##          B   17 2636    4    1    0
##          C    0    6 2376   14    0
##          D    0    0   25 2225    2
##          E    0    3    6    9 2507
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9916, 0.9945)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9913          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9957   0.9951   0.9847   0.9893   0.9988
## Specificity            0.9993   0.9980   0.9982   0.9976   0.9984
## Pos Pred Value         0.9982   0.9917   0.9917   0.9880   0.9929
## Neg Pred Value         0.9983   0.9988   0.9967   0.9979   0.9997
## Prevalence             0.2851   0.1928   0.1757   0.1637   0.1827
## Detection Rate         0.2838   0.1919   0.1730   0.1620   0.1825
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9975   0.9966   0.9915   0.9935   0.9986
```

```r
## out of bag error estimates
mean(fit2$finalModel$err.rate[,"OOB"])
```

```
## [1] 0.008421505
```

```r
## Cross-Validate Model with 2nd Training Set:
predictTrain2 <- predict(fit2, newdata=training2)
conMatrix <- confusionMatrix(predictTrain2, training2$classe)
print(conMatrix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    0 1135    4    0    0
##          C    1    2 1020    4    0
##          D    0    0    2  960    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9956, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9965   0.9942   0.9959   0.9991
## Specificity            0.9995   0.9992   0.9986   0.9994   1.0000
## Pos Pred Value         0.9988   0.9965   0.9932   0.9969   1.0000
## Neg Pred Value         0.9998   0.9992   0.9988   0.9992   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1929   0.1733   0.1631   0.1837
## Detection Prevalence   0.2846   0.1935   0.1745   0.1636   0.1837
## Balanced Accuracy      0.9995   0.9978   0.9964   0.9976   0.9995
```

## Plots of Training Data
![](Prediction_Assignment_files/figure-html/Plots-1.png)<!-- -->![](Prediction_Assignment_files/figure-html/Plots-2.png)<!-- -->![](Prediction_Assignment_files/figure-html/Plots-3.png)<!-- -->![](Prediction_Assignment_files/figure-html/Plots-4.png)<!-- -->


## Model Prediction Against Test Set


```r
## Read and Filter Test Data:
actyTestData <- read.csv("pml-testing.csv", header = TRUE)
actyTestData <- tbl_df(actyTestData)
testdata <- tbl_df(actyTestData)  ## Makes a copy

## Filter out columns to match Training Set
keepcolnamesT <-colnames(testdata)
names_vectorT <- !(grepl("^(kurtosis)",keepcolnamesT) | grepl("^(skewness)",keepcolnamesT) | grepl("^(max)",keepcolnamesT) | grepl("^(min)",keepcolnamesT) | grepl("^(amplitude)",keepcolnamesT)| grepl("^(var)",keepcolnamesT) | grepl("^(avg)",keepcolnamesT) | grepl("^(stddev)",keepcolnamesT) )

## Keep only Sensor Columns:
filtertestdata <- select(testdata,keepcolnamesT[names_vectorT])
filtertestdata <- select(filtertestdata,-c(1:7))

## Run Prediction on Test Set using model:
pred <- predict(fit2,filtertestdata)
pred <- tbl_df(pred)
pandoc.table(pred, style = "rmarkdown")
```

```
## 
## 
## | value |
## |:-----:|
## |   B   |
## |   A   |
## |   B   |
## |   A   |
## |   A   |
## |   E   |
## |   D   |
## |   B   |
## |   A   |
## |   A   |
## |   B   |
## |   C   |
## |   B   |
## |   A   |
## |   E   |
## |   E   |
## |   A   |
## |   B   |
## |   B   |
## |   B   |
```

```r
## Results Submitted to Quiz -- 100%!!
```
