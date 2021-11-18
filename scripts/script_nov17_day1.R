library(tidyverse)
library(mlr)


# get the data ------------------------------------------------------------

smoke_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl02/main/data/smoking.csv")

train_df <- smoke_df %>%  
  mutate(smoker = cigs > 0) %>% 
  dplyr::select(-cigs) %>% 
  relocate(smoker) %>% 
  as.data.frame()

# Make our "task" and "learner" -------------------------------------------

smokerTask <- makeClassifTask(data = train_df, target = 'smoker')

logReg <- makeLearner("classif.logreg", predict.type = "prob")

logRegTrained <- train(logReg, smokerTask)


# Evaluate predictive performance -----------------------------------------

p <- predict(logRegTrained, newdata = train_df)
calculateConfusionMatrix(p)
calculateROCMeasures(p)


# Get at the model --------------------------------------------------------

logReg_model <- getLearnerModel(logRegTrained)

# verify that this is just glm(...)
logReg_model_2 <- glm(smoker ~ ., data = train_df, family = binomial())


# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10,
                             stratify = TRUE)


logRegKfold <- mlr::resample(logReg, 
                             smokerTask, 
                             kfold_cv,
                             measures = list(acc, mmce, ppv, tpr, fpr, fdr, f1)
)

logRegKfold$aggr


# ROC and AUC -------------------------------------------------------------

roc_df <- generateThreshVsPerfData(p, 
                                   measures = list(fpr, tpr)
)

plotROCCurves(roc_df)
performance(p, measures = auc)



# naiive bayes classifier -------------------------------------------------

library(mlbench)
data("HouseVotes84")
head(HouseVotes84, 10)

votesTask <- makeClassifTask(data = HouseVotes84, target = "Class")
nbayesLearner <- makeLearner('classif.naiveBayes')
nbayesLearnerTrained <- train(nbayesLearner, votesTask)


# predictions -------------------------------------------------------------

p <- predict(nbayesLearnerTrained, newdata = HouseVotes84)
calculateROCMeasures(p)


# cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10, 
                             stratify = TRUE)

nbayesLearnerCV <- resample(learner = nbayesLearner, 
                            task = votesTask, 
                            resampling = kfold_cv,
                            measures = list(acc, mmce, ppv, tpr, fpr, fdr, f1)
)

nbayesLearnerCV$aggr


# Support vector machines -------------------------------------------------

library(kernlab)
data(spam)
as_tibble(spam)

spamTask <- makeClassifTask(data = spam, target = 'type')
svm_learner <- makeLearner('classif.svm', kernel = 'linear')
svm_trained <- train(svm_learner, spamTask)


# predictive performance --------------------------------------------------

p_linear <- predict(svm_trained, newdata = spam)
calculateConfusionMatrix(p_linear)
calculateROCMeasures(p_linear)

# Cross validation

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 10,
                             reps = 10,
                             stratify = TRUE)

svm_cv <- resample(learner = svm_learner,
                   task = spamTask,
                   resampling = kfold_cv,
                   measures = list(ppv, tpr, f1, fpr, fdr, acc, mmce)
)

svm_cv$aggr


# # optimize hyperparameters ----------------------------------------------

getParamSet(svm_learner)

# hyper-parameter space to search over
svm_param_space <- makeParamSet(
  makeNumericParam('cost', lower = 0.1, upper = 10),
  makeNumericParam('gamma', lower = 0.1, upper = 10)
)

rand_search <- makeTuneControlRandom(maxit = 25)

cv_for_tuning <- makeResampleDesc(method = 'Holdout', 
                                  split = 2/3)

svm_tuned <- tuneParams('classif.svm',
                        task = spamTask,
                        resampling = cv_for_tuning,
                        par.set = svm_param_space,
                        control = rand_search)

svm_tuned$x

svm_learner_tuned <- setHyperPars(makeLearner('classif.svm'), 
                                  par.vals = svm_tuned$x)

svm_learner_tuned_trained <- train(svm_learner_tuned, spamTask)

p_tuned <- predict(svm_learner_tuned_trained, newdata = spam)
calculateROCMeasures(p_tuned)
calculateConfusionMatrix(p_tuned)


svm_tuned_cv <- resample(learner = svm_learner_tuned,
                         task = spamTask,
                         resampling = kfold_cv,
                         measures = list(ppv, tpr, fpr, fdr, f1, acc, mmce)
)
    

svm_tuned_cv$aggr                     
