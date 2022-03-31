install.packages("pacman", quiet = T)
library(pacman)
pacman::p_load(
    pROC
)

# Output Plot for MLR
png("output/regression_mlr.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d1$G3, lm.fit.d1.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Mat w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, lm.fit.d1p.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Mat w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, lm.fit.d2.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Por w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, lm.fit.d2p.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Por w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Plot for KNN Regression
png("output/regression_knn.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d1$G3, knn.fit.d1.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Mat w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, knn.fit.d1p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Mat w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, knn.fit.d2.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Por w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, knn.fit.d2p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Por w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Plot for SVM Linear Regression
png("output/regression_svm_linear.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d1$G3, svm.linear.fit.d1.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Mat w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, svm.linear.fit.d1p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Mat w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.linear.fit.d2.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Por w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.linear.fit.d2p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Por w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Plot for SVM Radial Regression
png("output/regression_svm_radial.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d1$G3, svm.radial.fit.d1.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Mat w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, svm.radial.fit.d1p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Mat w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.radial.fit.d2.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Por w All", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.radial.fit.d2p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Por w Selected", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Plots for Regression: Mat
png("output/regression_mat.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d1$G3, lm.fit.d1p.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Mat", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, knn.fit.d1p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Mat", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, svm.linear.fit.d1.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Mat", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d1$G3, svm.radial.fit.d1.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Mat", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Plots for Regression: Por
png("output/regression_por.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1.2)
plot(d2$G3, lm.fit.d2p.pred, xlab = "Desired", ylab = "Predicted", main = "MLR on Por", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, knn.fit.d2p.reg.pred, xlab = "Desired", ylab = "Predicted", main = "KNN on Por", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.linear.fit.d2.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Linear on Por", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
plot(d2$G3, svm.radial.fit.d2.reg.pred, xlab = "Desired", ylab = "Predicted", main = "SVM Radial on Por", xlim = c(0,20), ylim = c(0,20))
abline(coef = c(0,1))
dev.off()

# Output Confusion Matrix for Classification: Accuracy
out <- capture.output(confusionMatrix(lda.fit.d1.pred, d1.cls$G3))
cat("####### LDA on Mat w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=FALSE)
out <- capture.output(confusionMatrix(lda.fit.d1p.pred, d1.cls$G3))
cat("\n\n####### LDA on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(lda.fit.d2.pred, d2.cls$G3))
cat("\n\n####### LDA on Por w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(lda.fit.d2p.pred, d2.cls$G3))
cat("\n\n####### LDA on Por w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d1.cls.pred, d1.cls$G3))
cat("\n\n####### KNN on Mat w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d1p.cls.pred, d1.cls$G3))
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d2.cls.pred, d2.cls$G3))
cat("\n\n####### KNN on Por w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d2p.cls.pred, d2.cls$G3))
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d1.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Linear on Mat w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d1p.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Linear on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d2.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d2p.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Linear on Por w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1p.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2p.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Radial on Por w Sel #######\n", out, file="output/classification_confusion_matrices_accuracy.txt", sep="\n", append=TRUE)

# Output Confusion Matrix for Classification: ROC
out <- capture.output(confusionMatrix(lda.fit.d1r.pred, d1.cls$G3))
cat("####### LDA on Mat w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=FALSE)
out <- capture.output(confusionMatrix(lda.fit.d1pr.pred, d1.cls$G3))
cat("\n\n####### LDA on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(lda.fit.d2r.pred, d2.cls$G3))
cat("\n\n####### LDA on Por w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(lda.fit.d2pr.pred, d2.cls$G3))
cat("\n\n####### LDA on Por w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d1r.cls.pred, d1.cls$G3))
cat("\n\n####### KNN on Mat w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d1pr.cls.pred, d1.cls$G3))
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d2r.cls.pred, d2.cls$G3))
cat("\n\n####### KNN on Por w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d2pr.cls.pred, d2.cls$G3))
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d1r.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Linear on Mat w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d1pr.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Linear on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d2r.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d2pr.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Linear on Por w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1r.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1pr.cls.pred, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2r.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2pr.cls.pred, d2.cls$G3))
cat("\n\n####### SVM Radial on Por w Sel #######\n", out, file="output/classification_confusion_matrices_roc.txt", sep="\n", append=TRUE)

# Dump each trained model detail
# MLR Regression model detail
out <- capture.output(lm.fit.d1)
cat("####### MLR on Mat w All #######\n", out, file="output/model_regression_MLR.txt", sep="\n", append=FALSE)
out <- capture.output(summary(lm.fit.d1))
cat(out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(lm.fit.d1p)
cat("####### MLR on Mat w Sel #######\n", out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(summary(lm.fit.d1p))
cat(out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(lm.fit.d2)
cat("\n\n####### MLR on Por w All #######\n", out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(summary(lm.fit.d2))
cat(out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(lm.fit.d2p)
cat("\n\n####### MLR on Por w Sel #######\n", out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)
out <- capture.output(summary(lm.fit.d2p))
cat(out, file="output/model_regression_MLR.txt", sep="\n", append=TRUE)

# KNN model detail
out <- capture.output(knn.fit.d1.reg)
cat("####### KNN on Mat w All #######\n", out, file="output/model_regression_KNN.txt", sep="\n", append=FALSE)
out <- capture.output(knn.fit.d1p.reg)
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/model_regression_KNN.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2.reg)
cat("\n\n####### KNN on Por w All #######\n", out, file="output/model_regression_KNN.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2p.reg)
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/model_regression_KNN.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d1.cls)
cat("####### KNN on Mat w All #######\n", out, file="output/model_classification_KNN_accuracy.txt", sep="\n", append=FALSE)
out <- capture.output(knn.fit.d1p.cls)
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/model_classification_KNN_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2.cls)
cat("\n\n####### KNN on Por w All #######\n", out, file="output/model_classification_KNN_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2p.cls)
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/model_classification_KNN_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d1r.cls)
cat("####### KNN on Mat w All #######\n", out, file="output/model_classification_KNN_roc.txt", sep="\n", append=FALSE)
out <- capture.output(knn.fit.d1pr.cls)
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/model_classification_KNN_roc.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2r.cls)
cat("\n\n####### KNN on Por w All #######\n", out, file="output/model_classification_KNN_roc.txt", sep="\n", append=TRUE)
out <- capture.output(knn.fit.d2pr.cls)
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/model_classification_KNN_roc.txt", sep="\n", append=TRUE)

# LDA model detail
out <- capture.output(lda.fit.d1)
cat("####### LDA on Mat w All #######\n", out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=FALSE)
out <- capture.output(lda.fit.d1$finalModel)
cat(out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d1p)
cat("\n\n####### LDA on Mat w Sel #######\n", out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d1p$finalModel)
cat(out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2)
cat("\n\n####### LDA on Por w All #######\n", out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2$finalModel)
cat(out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2p)
cat("\n\n####### LDA on Por w Sel #######\n", out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2p$finalModel)
cat(out, file="output/model_classification_LDA_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d1r)
cat("####### LDA on Mat w All #######\n", out, file="output/model_classification_LDA_roc.txt", sep="\n", append=FALSE)
out <- capture.output(lda.fit.d1r$finalModel)
cat(out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d1pr)
cat("\n\n####### LDA on Mat w Sel #######\n", out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d1pr$finalModel)
cat(out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2r)
cat("\n\n####### LDA on Por w All #######\n", out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2r$finalModel)
cat(out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2pr)
cat("\n\n####### LDA on Por w Sel #######\n", out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)
out <- capture.output(lda.fit.d2pr$finalModel)
cat(out, file="output/model_classification_LDA_roc.txt", sep="\n", append=TRUE)

# SVM Linear model detail regression
out <- capture.output(svm.linear.fit.d1.reg)
cat("####### SVM Linear on Mat w All #######\n", out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=FALSE)
out <- capture.output(svm.linear.fit.d1.reg$finalModel)
cat(out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1p.reg)
cat("\n\n####### SVM Linear on Mat w Sel #######\n", out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1p.reg$finalModel)
cat(out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2.reg)
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2.reg$finalModel)
cat(out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2p.reg)
cat("\n\n####### SVM Linear on Por w Sel #######\n", out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2p.reg$finalModel)
cat(out, file="output/model_regression_SVM_Linear.txt", sep="\n", append=TRUE)

# SVM Linear model detail classification
out <- capture.output(svm.linear.fit.d1.cls)
cat("####### SVM Linear on Mat w All #######\n", out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=FALSE)
out <- capture.output(svm.linear.fit.d1.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1p.cls)
cat("\n\n####### SVM Linear on Mat w Sel #######\n", out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1p.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2.cls)
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2p.cls)
cat("\n\n####### SVM Linear on Por w Sel #######\n", out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2p.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1r.cls)
cat("####### SVM Linear on Mat w All #######\n", out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=FALSE)
out <- capture.output(svm.linear.fit.d1r.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1pr.cls)
cat("\n\n####### SVM Linear on Mat w Sel #######\n", out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d1pr.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2r.cls)
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2r.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2pr.cls)
cat("\n\n####### SVM Linear on Por w Sel #######\n", out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.linear.fit.d2pr.cls$finalModel)
cat(out, file="output/model_classification_SVM_Linear_roc.txt", sep="\n", append=TRUE)

# SVM Radial model detail regression
out <- capture.output(svm.radial.fit.d1.reg)
cat("####### SVM Radial on Mat w All #######\n", out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=FALSE)
out <- capture.output(svm.radial.fit.d1.reg$finalModel)
cat(out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1p.reg)
cat("\n\n####### SVM Radial on Mat w Sel #######\n", out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1p.reg$finalModel)
cat(out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2.reg)
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2.reg$finalModel)
cat(out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2p.reg)
cat("\n\n####### SVM Radial on Por w Sel #######\n", out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2p.reg$finalModel)
cat(out, file="output/model_regression_SVM_Radial.txt", sep="\n", append=TRUE)

# SVM Radial model detail classification
out <- capture.output(svm.radial.fit.d1.cls)
cat("####### SVM Radial on Mat w All #######\n", out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=FALSE)
out <- capture.output(svm.radial.fit.d1.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1p.cls)
cat("\n\n####### SVM Radial on Mat w Sel #######\n", out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1p.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2.cls)
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2p.cls)
cat("\n\n####### SVM Radial on Por w Sel #######\n", out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2p.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_accuracy.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1r.cls)
cat("####### SVM Radial on Mat w All #######\n", out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=FALSE)
out <- capture.output(svm.radial.fit.d1r.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1pr.cls)
cat("\n\n####### SVM Radial on Mat w Sel #######\n", out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1pr.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2r.cls)
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2r.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2pr.cls)
cat("\n\n####### SVM Radial on Por w Sel #######\n", out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2pr.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_roc.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1rg.cls)
cat("####### SVM Radial on Mat wo G1 G2 #######\n", out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=FALSE)
out <- capture.output(svm.radial.fit.d1rg.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1rgg.cls)
cat("\n\n####### SVM Radial on Mat wo G1 #######\n", out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d1rgg.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2rg.cls)
cat("\n\n####### SVM Radial on Por wo G1 G2 #######\n", out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2rg.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2rgg.cls)
cat("\n\n####### SVM Radial on Por wo G1 #######\n", out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(svm.radial.fit.d2rgg.cls$finalModel)
cat(out, file="output/model_classification_SVM_Radial_bonus.txt", sep="\n", append=TRUE)

# ROC plot for LDA section
png("output/roc_lda.png", 800, 400, "px")
par(mfrow=c(1,2), cex=1)

# plot.roc(lda.fit.d1$pred$obs,lda.fit.d1$pred$Fail, col = 2, main = "ROC Curve: LDA on Mat", levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
plot.roc(lda.fit.d1$pred$obs,lda.fit.d1$pred$Fail, col = 2, main = "ROC Curve: LDA on Mat", levels = c("Fail","Pass"), direction=">")
plot.roc(lda.fit.d1p$pred$obs,lda.fit.d1p$pred$Fail, col = 3, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("All","Sel"), col = c(2,3), lwd =2)

plot.roc(lda.fit.d2$pred$obs,lda.fit.d2$pred$Fail, col = 2, main = "ROC Curve: LDA on Por", levels = c("Fail","Pass"), direction=">")
plot.roc(lda.fit.d2p$pred$obs,lda.fit.d2p$pred$Fail, col = 3, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("All","Sel"), col = c(2,3), lwd =2)

dev.off()

# ROC plot for KNN section
png("output/roc_knn.png", 800, 400, "px")
par(mfrow=c(1,2), cex=1)

selectedIndices <- knn.fit.d1p.cls$pred$k == 1
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: KNN on Mat w Sel\n (Best Acc k=41 ROC k=47)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 11
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 21
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 31
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 41
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 47
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("k=1","k=11","k=21","k=31","k=41","k=47"), col = c(2,3,4,5,6,7), lwd =2)

selectedIndices <- knn.fit.d2p.cls$pred$k == 1
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: KNN on Por w Sel\n (Best Acc k=17 ROC k=47)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 17
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 21
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 31
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 41
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 47
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("k=1","k=17","k=21","k=31","k=41","k=47"), col = c(2,3,4,5,6,7), lwd =2)

dev.off()

# ROC plot for SVM Linear section
png("output/roc_svm_linear.png", 800, 400, "px")
par(mfrow=c(1,2), cex=1)

selectedIndices <- svm.linear.fit.d1.cls$pred$C == 0.001
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Linear on Mat w All\n(Best C=1)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 0.01
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 10
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 100
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 1000
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 1
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

selectedIndices <- svm.linear.fit.d2.cls$pred$C == 0.001
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Linear on Por w All\n(Best C=0.1)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 0.01
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 1
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 10
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 100
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 1000
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

dev.off()

# ROC plot for SVM Radial section
png("output/roc_svm_radial.png", 800, 800, "px")
par(mfrow=c(2,2), cex=1)

selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 0.001) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Mat\n sigma=0.001 (Best C=1000)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 0.01) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 0.1) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 10) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 100) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Mat\n C=1000 (Best sigma=0.001)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.1)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 2, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("sigma=0.001","sigma=0.01","sigma=0.1"), col = c(2,3,4), lwd =2)

selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 0.001) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Por\n sigma=0.01 (Best C=10)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 0.01) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 0.1) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 1) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 100) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 1000) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Por\n C=10 (Best sigma=0.01)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.1)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("sigma=0.001","sigma=0.01","sigma=0.1"), col = c(2,3,4), lwd =2)

dev.off()

#ROC plot for summary
png("output/roc_summary.png", 800, 400, "px")

par(mfrow=c(1,2), cex=1)

plot.roc(lda.fit.d1p$pred$obs,lda.fit.d1p$pred$Fail, col = 2, main = "ROC Curve: Best of All Models on Mat", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 47
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1.cls$pred$C == 1
plot.roc(svm.linear.fit.d1.cls$pred$obs[selectedIndices],svm.linear.fit.d1.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("LDA","KNN","SVM Linear","SVM Radial"), col = c(2,3,4,5), lwd =2)

plot.roc(lda.fit.d2p$pred$obs,lda.fit.d2p$pred$Fail, col = 2, main = "ROC Curve: Best of All Models on Por", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 47
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d2.cls$pred$obs[selectedIndices],svm.linear.fit.d2.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("LDA","KNN","SVM Linear","SVM Radial"), col = c(2,3,4,5), lwd =2)

dev.off()

#ROC plot for bonus
png("output/roc_bonus.png", 800, 400, "px")

par(mfrow=c(1,2), cex=1)

selectedIndices <- (svm.radial.fit.d1rg.cls$pred$C == 1) & (svm.radial.fit.d1rg.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1rg.cls$pred$obs[selectedIndices],svm.radial.fit.d1rg.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: Mat Grade Availability", levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
selectedIndices <- (svm.radial.fit.d1rgg.cls$pred$C == 1000) & (svm.radial.fit.d1rgg.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1rgg.cls$pred$obs[selectedIndices],svm.radial.fit.d1rgg.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
selectedIndices <- (svm.radial.fit.d1.cls$pred$C == 1000) & (svm.radial.fit.d1.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1.cls$pred$obs[selectedIndices],svm.radial.fit.d1.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("wo G1 G2","wo G2","All"), col = c(2,3,4), lwd =2)

selectedIndices <- (svm.radial.fit.d2rg.cls$pred$C == 10) & (svm.radial.fit.d2rg.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2rg.cls$pred$obs[selectedIndices],svm.radial.fit.d2rg.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: Por Grade Availability", levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
selectedIndices <- (svm.radial.fit.d2rgg.cls$pred$C == 100) & (svm.radial.fit.d2rgg.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2rgg.cls$pred$obs[selectedIndices],svm.radial.fit.d2rgg.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
selectedIndices <- (svm.radial.fit.d2.cls$pred$C == 10) & (svm.radial.fit.d2.cls$pred$sigma == 0.01)
plot.roc(svm.radial.fit.d2.cls$pred$obs[selectedIndices],svm.radial.fit.d2.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">", print.thres="best", print.thres.best.method="closest.topleft")
legend("bottomright", legend = c("wo G1 G2","wo G2","All"), col = c(2,3,4), lwd =2)

dev.off()

#ROC plot for mat
png("output/roc_mat.png", 800, 800, "px")

par(mfrow=c(2,2), cex=1)

selectedIndices <- knn.fit.d1p.cls$pred$k == 1
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: KNN on Mat (Best k=41)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 11
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 21
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 31
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 41
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 47
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("k=1","k=11","k=21","k=31","k=41","k=49"), col = c(2,3,4,5,6,7), lwd =2)

selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 0.001
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Linear on Mat (Best C=1)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 0.01
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 1
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 10
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 100
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 1000
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 0.001) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Mat (Best C=1000)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 0.01) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 0.1) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 1) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 10) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 100) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 1000) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

plot.roc(lda.fit.d1p$pred$obs,lda.fit.d1p$pred$Fail, col = 2, main = "ROC Curve: Best of All Models on Mat", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d1p.cls$pred$k == 41
plot.roc(knn.fit.d1p.cls$pred$obs[selectedIndices],knn.fit.d1p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d1p.cls$pred$C == 1
plot.roc(svm.linear.fit.d1p.cls$pred$obs[selectedIndices],svm.linear.fit.d1p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d1p.cls$pred$C == 1000) & (svm.radial.fit.d1p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d1p.cls$pred$obs[selectedIndices],svm.radial.fit.d1p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("LDA","KNN","SVM Linear","SVM Radial"), col = c(2,3,4,5), lwd =2)

dev.off()

#ROC plot for Por
png("output/roc_por.png", 800, 800, "px")

par(mfrow=c(2,2), cex=1)

selectedIndices <- knn.fit.d2p.cls$pred$k == 1
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: KNN on Por (Best k=1)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 11
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 21
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 31
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 41
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 47
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("k=1","k=11","k=21","k=31","k=41","k=49"), col = c(2,3,4,5,6,7), lwd =2)

selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 0.001
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Linear on Por (Best C=0.1)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 0.01
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 1
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 10
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 100
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 1000
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 0.001) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 2, main = "ROC Curve: SVM Radial on Por (Best C=10)", levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 0.01) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 0.1) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 1) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 10) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 6, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 100) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 7, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 1000) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 8, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("C=0.001","C=0.01","C=0.1","C=1","C=10","C=100","C=1000"), col = c(2,3,4,5,6,7,8), lwd =2)

plot.roc(lda.fit.d2p$pred$obs,lda.fit.d2p$pred$Fail, col = 2, main = "ROC Curve: Best of All Models on Por", levels = c("Fail","Pass"), direction=">")
selectedIndices <- knn.fit.d2p.cls$pred$k == 1
plot.roc(knn.fit.d2p.cls$pred$obs[selectedIndices],knn.fit.d2p.cls$pred$Fail[selectedIndices], col = 3, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- svm.linear.fit.d2p.cls$pred$C == 0.1
plot.roc(svm.linear.fit.d2p.cls$pred$obs[selectedIndices],svm.linear.fit.d2p.cls$pred$Fail[selectedIndices], col = 4, add = T, levels = c("Fail","Pass"), direction=">")
selectedIndices <- (svm.radial.fit.d2p.cls$pred$C == 10) & (svm.radial.fit.d2p.cls$pred$sigma == 0.001)
plot.roc(svm.radial.fit.d2p.cls$pred$obs[selectedIndices],svm.radial.fit.d2p.cls$pred$Fail[selectedIndices], col = 5, add = T, levels = c("Fail","Pass"), direction=">")
legend("bottomright", legend = c("LDA","KNN","SVM Linear","SVM Radial"), col = c(2,3,4,5), lwd =2)

dev.off()

# Output Confusion Matrix for Classification: final
out <- capture.output(confusionMatrix(lda.fit.d1p.pred.thres, d1.cls$G3))
cat("####### LDA on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=FALSE)
out <- capture.output(confusionMatrix(lda.fit.d2p.pred.thres, d2.cls$G3))
cat("\n\n####### LDA on Por w Sel #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d1pr.cls.pred.thres, d1.cls$G3))
cat("\n\n####### KNN on Mat w Sel #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(knn.fit.d2pr.cls.pred.thres, d2.cls$G3))
cat("\n\n####### KNN on Por w Sel #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d1r.cls.pred.thres, d1.cls$G3))
cat("\n\n####### SVM Linear on Mat w All #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.linear.fit.d2r.cls.pred.thres, d2.cls$G3))
cat("\n\n####### SVM Linear on Por w All #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1r.cls.pred.thres, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat w All #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2r.cls.pred.thres, d2.cls$G3))
cat("\n\n####### SVM Radial on Por w All #######\n", out, file="output/classification_confusion_matrices_final.txt", sep="\n", append=TRUE)

# Output Confusion Matrix for Classification: bonus
out <- capture.output(confusionMatrix(svm.radial.fit.d1rg.cls.pred.thres, d1.cls$G3))
cat("####### SVM Radial on Mat wo G1 G2 #######\n", out, file="output/classification_confusion_matrices_bonus.txt", sep="\n", append=FALSE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2rg.cls.pred.thres, d2.cls$G3))
cat("\n\n####### SVM Radial on Por wo G1 G2 #######\n", out, file="output/classification_confusion_matrices_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d1rgg.cls.pred.thres, d1.cls$G3))
cat("\n\n####### SVM Radial on Mat wo G2 #######\n", out, file="output/classification_confusion_matrices_bonus.txt", sep="\n", append=TRUE)
out <- capture.output(confusionMatrix(svm.radial.fit.d2rgg.cls.pred.thres, d2.cls$G3))
cat("\n\n####### SVM Radial on Por wo G2 #######\n", out, file="output/classification_confusion_matrices_bonus.txt", sep="\n", append=TRUE)

# Plot histgram to show positive sample imbalance nature of each datasets
png("output/bar_data_distribution.png", 800, 400, "px")
par(mfrow=c(1,2), cex=1)
barplot(table(d1.cls$G3), main = "Mathematics")
barplot(table(d2.cls$G3), main = "Portugese")
dev.off()

# Plot Pairwise scatter plot for selected predictors for each dataset
png("output/pair_mat.png", 800, 800, "px")
pairs(~G3 + age + activities + famrel + absences + G1 + G2, data = d1, main = "Mathematics Selected Predictor")
dev.off()
png("output/pair_por.png", 800, 800, "px")
pairs(~G3 + Fjob + reason + traveltime + failures + G1 + G2, data = d2, main = "Portugese Selected Predictor")
dev.off()