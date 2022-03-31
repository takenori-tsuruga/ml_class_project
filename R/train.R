install.packages("pacman", quiet = T)
library(pacman)
pacman::p_load(
    caret,
    MASS,
    kernlab,
    doParallel,
    pROC
)

# Import data set
d1=read.table("student/student-mat.csv",sep=";",header=TRUE)
d2=read.table("student/student-por.csv",sep=";",header=TRUE)

# Listing factor (binary, nomial) columns for conversion
col.factor <- list(
    "sex",
    "school",
    "address",
    "Pstatus",
    "Mjob",
    "Fjob",
    "guardian",
    "famsize",
    "reason",
    "schoolsup",
    "famsup",
    "activities",
    "paid",
    "internet",
    "nursery",
    "higher",
    "romantic"
)

# Listing numerical columns for standardization and scaling
col.num <- list(
    "age",
    "Medu",
    "Fedu",
    "famrel",
    "traveltime",
    "studytime",
    "failures",
    "freetime",
    "goout",
    "Walc",
    "Dalc",
    "health",
    "absences",
    "G1",
    "G2"
)

# Convert factor columns from character
for (i in col.factor) { 
    d1[,i] <- as.factor(d1[,i])
    d2[,i] <- as.factor(d2[,i])
}

# Apply PCA process on numerical columns
for (i in col.num) { 
    d1[,i] <- prcomp(d1[,i], center=TRUE, scale=TRUE)$x
    d2[,i] <- prcomp(d2[,i], center=TRUE, scale=TRUE)$x
}

# Create data set for classification training
d1.cls <- d1 
for (i in 1: 395) {
    if (d1[i,"G3"] <  10 ) { d1.cls[i,"G3"] <- "Fail" }
    else { d1.cls[i,"G3"] <- "Pass" }
}
d1.cls[,"G3"] <- as.factor(d1.cls[,"G3"])

d2.cls <- d2
for (i in 1: 649) {
    if (d2[i,"G3"] <  10 ) { d2.cls[i,"G3"] <- "Fail" }
    else { d2.cls[i,"G3"] <- "Pass" }
}
d2.cls[,"G3"] <- as.factor(d2.cls[,"G3"])

# Selected predictors based on p-value from lm() outcome p < 0.1
# d1 "age + activities + famrel + absences + G1 + G2"
# d2 "Fjob + reason + traveltime + failures + G1 + G2"

# Initialize parallel processing (doParallel) with cluster number 12
cl <- makePSOCKcluster(12)
registerDoParallel(cl)

# set trControl value for tain(), K fold corss validation sampling with K=10
ctrl <- trainControl(method = "cv", number = 10)

# Training Multiple Linear Regression Model for regression
set.seed(1234)
lm.fit.d1 <- train(G3~., data=d1, method = "lm", trControl = ctrl)
set.seed(1234)
lm.fit.d2 <- train(G3~., data=d2, method = "lm", trControl = ctrl)
set.seed(1234)
lm.fit.d1p <- train(G3~age + activities + famrel + absences + G1 + G2, data=d1, method = "lm", trControl = ctrl)
set.seed(1234)
lm.fit.d2p <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, data=d2, method = "lm", trControl = ctrl)

# Training K-Nearest Neighbors Model for regression with K=(1,3,5,...,49)
tuneGrid.knn <- data.frame (
    .k = c(seq(1,49,2))
)
set.seed(1234)
knn.fit.d1.reg <- train(G3~., tuneGrid = tuneGrid.knn, data=d1, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d2.reg <- train(G3~., tuneGrid = tuneGrid.knn, data=d2, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d1p.reg <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.knn, data=d1, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d2p.reg <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.knn, data=d2, method = "knn", trControl = ctrl)

# Training Support Vector Machine Model with Linear Kernel for regression with C=(1e-03~1e+03)
tuneGrid.svm.linear <- data.frame (
    .C = c(
        1e-03,
        1e-02,
        1e-01,
        1e+00,
        1e+01,
        1e+02,
        1e+03
    )
)
set.seed(1234)
svm.linear.fit.d1.reg <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d1, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d2.reg <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d2, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d1p.reg <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d1, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d2p.reg <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d2, method = "svmLinear", trControl = ctrl)
    
# Training Support Vector Machine Model with Radial Kernel for regression with C=(1e-03~1e+03) sigma=1e-03
tuneGrid.svm.radial <- data.frame (
    .sigma = c(
        rep(c(
            1e-03,
            1e-02,
            1e-01,
            1e+00,
            1e+01,
            1e+02,
            1e+03),
            each = 7)
    ),
    .C = c(
        rep(c(
            1e-03,
            1e-02,
            1e-01,
            1e+00,
            1e+01,
            1e+02,
            1e+03),
            times = 7)
    )
)
set.seed(1234)
svm.radial.fit.d1.reg <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d1, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d2.reg <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d2, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d1p.reg <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d1, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d2p.reg <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d2, method = "svmRadial", trControl = ctrl)

# set trControl value for tain(), K fold corss validation sampling with K=10 
ctrl <- trainControl(method = "cv", number = 10, savePredictions = T, classProbs = T)

# Training K-Nearest Neighbors Model for classification with K=(1,3,5,...,49)
set.seed(1234)
knn.fit.d1.cls <- train(G3~., tuneGrid = tuneGrid.knn, data=d1.cls, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d2.cls <- train(G3~., tuneGrid = tuneGrid.knn, data=d2.cls, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d1p.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.knn, data=d1.cls, method = "knn", trControl = ctrl)
set.seed(1234)
knn.fit.d2p.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.knn, data=d2.cls, method = "knn", trControl = ctrl)

# Training Linear Discriminant Analysis Model for classification
set.seed(1234)
lda.fit.d1 <- train(G3~., data=d1.cls, method = "lda", trControl = ctrl)
set.seed(1234)
lda.fit.d2 <- train(G3~., data=d2.cls, method = "lda", trControl = ctrl)
set.seed(1234)
lda.fit.d1p <- train(G3~age + activities + famrel + absences + G1 + G2, data=d1.cls, method = "lda", trControl = ctrl)
set.seed(1234)
lda.fit.d2p <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, data=d2.cls, method = "lda", trControl = ctrl)

# Training Support Vector Machine Model with Linear Kernel for classification with C=(1e-03~1e+03)
set.seed(1234)
svm.linear.fit.d1.cls <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d1.cls, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d2.cls <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d2.cls, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d1p.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d1.cls, method = "svmLinear", trControl = ctrl)
set.seed(1234)
svm.linear.fit.d2p.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d2.cls, method = "svmLinear", trControl = ctrl)

# Training Support Vector Machine Model with Radial Kernel for both regression and classification with C=(1e-03~1e+03) sigma=1e-03
set.seed(1234)
svm.radial.fit.d1.cls <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d2.cls <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d1p.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl)
set.seed(1234)
svm.radial.fit.d2p.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl)

# set trControl value for tain(), K fold corss validation sampling with K=10 with ROC config 
ctrl <- trainControl(method = "cv", number = 10, savePredictions = T, classProbs = T, summaryFunction = twoClassSummary)

# Training K-Nearest Neighbors Model for classification with K=(1,3,5,...,49)
set.seed(1234)
knn.fit.d1r.cls <- train(G3~., tuneGrid = tuneGrid.knn, data=d1.cls, method = "knn", trControl = ctrl, metric = "ROC")
set.seed(1234)
knn.fit.d2r.cls <- train(G3~., tuneGrid = tuneGrid.knn, data=d2.cls, method = "knn", trControl = ctrl, metric = "ROC")
set.seed(1234)
knn.fit.d1pr.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.knn, data=d1.cls, method = "knn", trControl = ctrl, metric = "ROC")
set.seed(1234)
knn.fit.d2pr.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.knn, data=d2.cls, method = "knn", trControl = ctrl, metric = "ROC")

# Training Linear Discriminant Analysis Model for classification
set.seed(1234)
lda.fit.d1r <- train(G3~., data=d1.cls, method = "lda", trControl = ctrl, metric = "ROC")
set.seed(1234)
lda.fit.d2r <- train(G3~., data=d2.cls, method = "lda", trControl = ctrl, metric = "ROC")
set.seed(1234)
lda.fit.d1pr <- train(G3~age + activities + famrel + absences + G1 + G2, data=d1.cls, method = "lda", trControl = ctrl, metric = "ROC")
set.seed(1234)
lda.fit.d2pr <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, data=d2.cls, method = "lda", trControl = ctrl, metric = "ROC")

# Training Support Vector Machine Model with Linear Kernel for classification with C=(1e-03~1e+03)
set.seed(1234)
svm.linear.fit.d1r.cls <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d1.cls, method = "svmLinear", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.linear.fit.d2r.cls <- train(G3~., tuneGrid = tuneGrid.svm.linear, data=d2.cls, method = "svmLinear", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.linear.fit.d1pr.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d1.cls, method = "svmLinear", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.linear.fit.d2pr.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.linear, data=d2.cls, method = "svmLinear", trControl = ctrl, metric = "ROC")

# Training Support Vector Machine Model with Radial Kernel for both regression and classification with C=(1e-03~1e+03) sigma=1e-03
set.seed(1234)
svm.radial.fit.d1r.cls <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d2r.cls <- train(G3~., tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d1pr.cls <- train(G3~age + activities + famrel + absences + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d2pr.cls <- train(G3~Fjob + reason + traveltime + failures + G1 + G2, tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d1rg.cls <- train(G3~.- G1 - G2, tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d2rg.cls <- train(G3~.- G1 - G2, tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d1rgg.cls <- train(G3~.- G2, tuneGrid = tuneGrid.svm.radial, data=d1.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")
set.seed(1234)
svm.radial.fit.d2rgg.cls <- train(G3~.- G2, tuneGrid = tuneGrid.svm.radial, data=d2.cls, method = "svmRadial", trControl = ctrl, metric = "ROC")


# End doCluster prallel processing
stopCluster(cl)

# Make prediction with each model
lm.fit.d1.pred <- predict(lm.fit.d1, d1)
lm.fit.d2.pred <- predict(lm.fit.d2, d2)
lm.fit.d1p.pred <- predict(lm.fit.d1p, d1)
lm.fit.d2p.pred <- predict(lm.fit.d2p, d2)
knn.fit.d1.reg.pred <- predict(knn.fit.d1.reg, d1)
knn.fit.d2.reg.pred <- predict(knn.fit.d2.reg, d2)
knn.fit.d1p.reg.pred <- predict(knn.fit.d1p.reg, d1)
knn.fit.d2p.reg.pred <- predict(knn.fit.d2p.reg, d2)
knn.fit.d1.cls.pred <- predict(knn.fit.d1.cls, d1.cls)
knn.fit.d2.cls.pred <- predict(knn.fit.d2.cls, d2.cls)
knn.fit.d1p.cls.pred <- predict(knn.fit.d1p.cls, d1.cls)
knn.fit.d2p.cls.pred <- predict(knn.fit.d2p.cls, d2.cls)
knn.fit.d1r.cls.pred <- predict(knn.fit.d1r.cls, d1.cls)
knn.fit.d2r.cls.pred <- predict(knn.fit.d2r.cls, d2.cls)
knn.fit.d1pr.cls.pred <- predict(knn.fit.d1pr.cls, d1.cls)
knn.fit.d2pr.cls.pred <- predict(knn.fit.d2pr.cls, d2.cls)
lda.fit.d1.pred <- predict(lda.fit.d1, d1.cls)
lda.fit.d2.pred <- predict(lda.fit.d2, d2.cls)
lda.fit.d1p.pred <- predict(lda.fit.d1p, d1.cls)
lda.fit.d2p.pred <- predict(lda.fit.d2p, d2.cls)
lda.fit.d1r.pred <- predict(lda.fit.d1r, d1.cls)
lda.fit.d2r.pred <- predict(lda.fit.d2r, d2.cls)
lda.fit.d1pr.pred <- predict(lda.fit.d1pr, d1.cls)
lda.fit.d2pr.pred <- predict(lda.fit.d2pr, d2.cls)
svm.linear.fit.d1.reg.pred <- predict(svm.linear.fit.d1.reg, d1)
svm.linear.fit.d2.reg.pred <- predict(svm.linear.fit.d2.reg, d2)
svm.linear.fit.d1p.reg.pred <- predict(svm.linear.fit.d1p.reg, d1)
svm.linear.fit.d2p.reg.pred <- predict(svm.linear.fit.d2p.reg, d2)
svm.linear.fit.d1.cls.pred <- predict(svm.linear.fit.d1.cls, d1.cls)
svm.linear.fit.d2.cls.pred <- predict(svm.linear.fit.d2.cls, d2.cls)
svm.linear.fit.d1p.cls.pred <- predict(svm.linear.fit.d1p.cls, d1.cls)
svm.linear.fit.d2p.cls.pred <- predict(svm.linear.fit.d2p.cls, d2.cls)
svm.linear.fit.d1r.cls.pred <- predict(svm.linear.fit.d1r.cls, d1.cls)
svm.linear.fit.d2r.cls.pred <- predict(svm.linear.fit.d2r.cls, d2.cls)
svm.linear.fit.d1pr.cls.pred <- predict(svm.linear.fit.d1pr.cls, d1.cls)
svm.linear.fit.d2pr.cls.pred <- predict(svm.linear.fit.d2pr.cls, d2.cls)
svm.radial.fit.d1.reg.pred <- predict(svm.radial.fit.d1.reg, d1)
svm.radial.fit.d2.reg.pred <- predict(svm.radial.fit.d2.reg, d2)
svm.radial.fit.d1p.reg.pred <- predict(svm.radial.fit.d1p.reg, d1)
svm.radial.fit.d2p.reg.pred <- predict(svm.radial.fit.d2p.reg, d2)
svm.radial.fit.d1.cls.pred <- predict(svm.radial.fit.d1.cls, d1.cls)
svm.radial.fit.d2.cls.pred <- predict(svm.radial.fit.d2.cls, d2.cls)
svm.radial.fit.d1p.cls.pred <- predict(svm.radial.fit.d1p.cls, d1.cls)
svm.radial.fit.d2p.cls.pred <- predict(svm.radial.fit.d2p.cls, d2.cls)
svm.radial.fit.d1r.cls.pred <- predict(svm.radial.fit.d1r.cls, d1.cls)
svm.radial.fit.d2r.cls.pred <- predict(svm.radial.fit.d2r.cls, d2.cls)
svm.radial.fit.d1pr.cls.pred <- predict(svm.radial.fit.d1pr.cls, d1.cls)
svm.radial.fit.d2pr.cls.pred <- predict(svm.radial.fit.d2pr.cls, d2.cls)
svm.radial.fit.d1rg.cls.pred <- predict(svm.radial.fit.d1rg.cls, d1.cls)
svm.radial.fit.d2rg.cls.pred <- predict(svm.radial.fit.d2rg.cls, d2.cls)
svm.radial.fit.d1rgg.cls.pred <- predict(svm.radial.fit.d1rgg.cls, d1.cls)
svm.radial.fit.d2rgg.cls.pred <- predict(svm.radial.fit.d2rgg.cls, d2.cls)


# Data processing for threshold prediction
lda.fit.d1p.pred.prob <- predict(lda.fit.d1p, d1.cls, type = "prob")
lda.fit.d2p.pred.prob <- predict(lda.fit.d2p, d2.cls, type = "prob")
knn.fit.d1pr.cls.pred.prob <- predict(knn.fit.d1pr.cls, d1.cls, type = "prob")
knn.fit.d2pr.cls.pred.prob <- predict(knn.fit.d2pr.cls, d2.cls, type = "prob")
svm.linear.fit.d1r.cls.pred.prob <- predict(svm.linear.fit.d1r.cls, d1.cls, type = "prob")
svm.linear.fit.d2r.cls.pred.prob <- predict(svm.linear.fit.d2r.cls, d2.cls, type = "prob")
svm.radial.fit.d1r.cls.pred.prob <- predict(svm.radial.fit.d1r.cls, d1.cls, type = "prob")
svm.radial.fit.d2r.cls.pred.prob <- predict(svm.radial.fit.d2r.cls, d2.cls, type = "prob")
svm.radial.fit.d1rg.cls.pred.prob <- predict(svm.radial.fit.d1rg.cls, d1.cls, type = "prob")
svm.radial.fit.d2rg.cls.pred.prob <- predict(svm.radial.fit.d2rg.cls, d2.cls, type = "prob")
svm.radial.fit.d1rgg.cls.pred.prob <- predict(svm.radial.fit.d1rgg.cls, d1.cls, type = "prob")
svm.radial.fit.d2rgg.cls.pred.prob <- predict(svm.radial.fit.d2rgg.cls, d2.cls, type = "prob")

lda.fit.d1p.pred.thres <- factor(,levels = c("Fail","Pass"))
lda.fit.d2p.pred.thres <- factor(,levels = c("Fail","Pass"))
knn.fit.d1pr.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
knn.fit.d2pr.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.linear.fit.d1r.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.linear.fit.d2r.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d1r.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d2r.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d1rg.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d2rg.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d1rgg.cls.pred.thres <- factor(,levels = c("Fail","Pass"))
svm.radial.fit.d2rgg.cls.pred.thres <- factor(,levels = c("Fail","Pass"))

for (i in 1:395) { if (lda.fit.d1p.pred.prob[i,1] >= 0.388 ) { lda.fit.d1p.pred.thres[i] <- "Fail"} else { lda.fit.d1p.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (lda.fit.d2p.pred.prob[i,1] >= 0.130 ) { lda.fit.d2p.pred.thres[i] <- "Fail"} else { lda.fit.d2p.pred.thres[i] <- "Pass" }}
for (i in 1:395) { if (knn.fit.d1pr.cls.pred.prob[i,1] >= 0.337 ) { knn.fit.d1pr.cls.pred.thres[i] <- "Fail"} else { knn.fit.d1pr.cls.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (knn.fit.d2pr.cls.pred.prob[i,1] >= 0.173 ) { knn.fit.d2pr.cls.pred.thres[i] <- "Fail"} else { knn.fit.d2pr.cls.pred.thres[i] <- "Pass" }}
for (i in 1:395) { if (svm.linear.fit.d1r.cls.pred.prob[i,1] >= 0.412 ) { svm.linear.fit.d1r.cls.pred.thres[i] <- "Fail"} else { svm.linear.fit.d1r.cls.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (svm.linear.fit.d2r.cls.pred.prob[i,1] >= 0.175 ) { svm.linear.fit.d2r.cls.pred.thres[i] <- "Fail"} else { svm.linear.fit.d2r.cls.pred.thres[i] <- "Pass" }}
for (i in 1:395) { if (svm.radial.fit.d1r.cls.pred.prob[i,1] >= 0.435 ) { svm.radial.fit.d1r.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d1r.cls.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (svm.radial.fit.d2r.cls.pred.prob[i,1] >= 0.190 ) { svm.radial.fit.d2r.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d2r.cls.pred.thres[i] <- "Pass" }}
for (i in 1:395) { if (svm.radial.fit.d1rg.cls.pred.prob[i,1] >= 0.291 ) { svm.radial.fit.d1rg.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d1rg.cls.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (svm.radial.fit.d2rg.cls.pred.prob[i,1] >= 0.171 ) { svm.radial.fit.d2rg.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d2rg.cls.pred.thres[i] <- "Pass" }}
for (i in 1:395) { if (svm.radial.fit.d1rgg.cls.pred.prob[i,1] >= 0.332 ) { svm.radial.fit.d1rgg.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d1rgg.cls.pred.thres[i] <- "Pass" }}
for (i in 1:649) { if (svm.radial.fit.d2rgg.cls.pred.prob[i,1] >= 0.214 ) { svm.radial.fit.d2rgg.cls.pred.thres[i] <- "Fail"} else { svm.radial.fit.d2rgg.cls.pred.thres[i] <- "Pass" }}
