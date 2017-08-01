## Inits
libs <- c("tm","caret","doParallel","MLmetrics","qdap","caretEnsemble","stringr")
lapply(libs, require, character.only=TRUE)
setwd("/...")

## Download and read data
download.file(url="http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/smsspamcollection.zip",
              destfile="./spam-detection/smsspamcollection.zip", method="curl")


spam_ham_data <- read.table(unz("./spam-detection/smsspamcollection.zip","SMSSpamCollection.txt"), 
                            header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)

## Prep rectangular data
colnames(spam_ham_data) <- c("type","message")
# spam_ham_data$type <- as.factor(ifelse(spam_ham_data$type == "ham", 1, 0))
spam_ham_data$type <- as.factor(spam_ham_data$type)
spam_ham_data$message<- as.character(spam_ham_data$message)

## Most frequent words (100 with length 4)
freq_words <-  unname(unlist((freq_terms(spam_ham_data,top = 100, at.least = 4, stopwords = tm::stopwords("english")))["WORD"]))

## Word count of message
word_cnt <- str_count(spam_ham_data$message," ") + 1

## Character count of message
char_cnt <- nchar(spam_ham_data$message)

## Vector of spam strings
spam_strings <- c("[0-9]{4}","[A-Z]{4}","£","\\!\\!","\\,","\\.\\.\\.","www","xxx")

## Feature vector
features <- c(spam_strings, freq_words)

## Develop analysis data frame
analysis_data <- as.data.frame(cbind(word_cnt,char_cnt,spam_ham_data$type))

for (feature in features) {
  analysis_data <- cbind(as.numeric(grepl(feature, spam_ham_data$message)),analysis_data)
}

names(analysis_data) <- c("numbers","upper_case","pounds","exclMark","comma","suspPoints",
                          "www","xxx",freq_words,"word_cnt","char_cnt","spam")

analysis_data$spam <- as.factor(ifelse(analysis_data$spam == 1, "ham","spam"))

## Split data
set.seed(2112)
idx <- createDataPartition(analysis_data$spam, p=0.8, list=FALSE)
train <- analysis_data[idx,]
test <- analysis_data[-idx,]

## Control function
cvCtrl <- trainControl(method = "repeatedcv",     
                       number = 5,							  
                       summaryFunction=prSummary,	# allow precision as metric
                       classProbs=TRUE,
                       savePredictions=TRUE)           

## Model assignment
model <- spam ~ .

## Parallel processing
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

## Random Forest
rf_fit <- train(spam ~ ., 
                data = train, 
                method="ranger",
                metric="Recall",
                trControl=cvCtrl)

## Performance
plot(rf_fit)  	
rf_pred <- predict(rf_fit,test)
confusionMatrix(rf_pred,test$spam)  



## XgBoost
xgb_fit <- train(spam ~ ., 
                 data = train, 
                 method="xgbTree",
                 metric="Recall",
                 trControl=cvCtrl)

## Performance
plot(xgb_fit) 
xgb_pred <- predict(xgb_fit,test)
confusionMatrix(xgb_pred,test$spam)  



## MDA
mda_fit <- train(spam ~ ., 
                 data = train, 
                 method="mda",
                 metric="Recall",
                 trControl=cvCtrl)

## Performance
plot(mda_fit) 
mda_pred <- predict(mda_fit,test)
confusionMatrix(mda_pred,test$spam) 



## MARS
mars_fit <- train(spam ~ ., 
                  data = train, 
                  method="earth",
                  metric="Recall",
                  trControl=cvCtrl)

## Performance
plot(mars_fit) 
mars_pred <- predict(mars_fit,test)
confusionMatrix(mars_pred,test$spam) 



## SVM
svm_fit <- train(spam ~ ., 
                 data = train, 
                 method="svmRadial",
                 metric="Recall",
                 trControl=cvCtrl)

## Performance
plot(svm_fit) 
svm_pred <- predict(svm_fit,test)
confusionMatrix(svm_pred,test$spam)



## KNN
knn_fit <- train(spam ~ ., 
                 data = train, 
                 method="knn",
                 metric="Recall",
                 trControl=cvCtrl)

## Performance
plot(knn_fit) 
knn_pred <- predict(knn_fit,test)
confusionMatrix(knn_pred,test$spam)



## NN
nn_fit <- train(spam ~ ., 
                data = train, 
                method="nnet",
                metric="Recall",
                trControl=cvCtrl)

## Performance
plot(nn_fit) 
nn_pred <- predict(nn_fit,test)
confusionMatrix(nn_pred,test$spam)



## C5.0
c50_fit <- train(spam ~ ., 
                 data = train, 
                 method="C5.0",
                 metric="Recall",
                 trControl=cvCtrl)

## Performance
plot(c50_fit) 
c50_pred <- predict(c50_fit,test)
confusionMatrix(c50_pred,test$spam)




## Ensemble
set.seed(2112)
algorithmList <- c('nnet','knn','svmRadial','mda')


models <- caretList(model, 
                    data=train, 
                    trControl=cvCtrl, 
                    methodList=algorithmList)

## Performance
results <- resamples(models)
summary(results)
dotplot(results)

# Correlation between results
modelCor(results)
splom(results)


# stack using rf
set.seed(2112)
stack_fit <- caretStack(models, 
                        method="ranger", 
                        metric="Recall", 
                        trControl=cvCtrl)
print(stack_fit)

## Performance
stack_pred <- predict(stack_fit,test)
confusionMatrix(stack_pred,test$spam)

