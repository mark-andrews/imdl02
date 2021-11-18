library(tidyverse)
library(mlr)
library(mlbench)

data(Zoo)

Zoo %>% as_tibble()

group_by(Zoo, type) %>% summarize(n = n())

Zoo <- mutate(Zoo, 
              across(where(is.logical), as.factor)
)

# Decision tree classifier ------------------------------------------------

ZooTask <- makeClassifTask(data = Zoo, target = 'type')
dtree <- makeLearner('classif.rpart')
dtree_trained <- train(dtree, ZooTask)

# Visualize tree ----------------------------------------------------------

library(rpart.plot)
dtree_model <- getLearnerModel(dtree_trained)
rpart.plot(dtree_model)


# Evaluate performance ----------------------------------------------------

p_dtree_default <- predict(dtree_trained, newdata = Zoo)
calculateConfusionMatrix(p_dtree_default)
performance(p_dtree_default, measures = list(acc, mmce))

kfold_cv <- makeResampleDesc(method = 'RepCV', folds = 10, reps = 10)

dtree_default_cv <- resample(learner = dtree,
                             task = ZooTask,
                             resampling = kfold_cv,
                             measures = list(acc, mmce))

dtree_default_cv$aggr      


# optimize hyper-parameters -----------------------------------------------

getParamSet(dtree)

dtree_param_space <- makeParamSet(
  makeIntegerParam('minsplit', lower = 3, upper = 25),
  makeIntegerParam('minbucket', lower = 3, upper = 25),
  makeIntegerParam('maxdepth', lower = 3, upper = 20),
  makeNumericParam('cp', lower = 0.001, upper = 0.5)
)

rand_search <- makeTuneControlRandom(maxit = 250)

cv_for_dtree_tuning <- makeResampleDesc(method = 'RepCV',
                                        folds = 10, 
                                        reps = 2)

dtree_tuned_params <- tuneParams(learner = dtree,
                                 task = ZooTask,
                                 resampling = cv_for_dtree_tuning,
                                 par.set = dtree_param_space,
                                 control = rand_search)

dtree_tuned_params$x           


# set params to their optimal values
dtree_tuned <- setHyperPars(dtree, par.vals = dtree_tuned_params$x)

dtree_tuned_cv <- resample(learner = dtree_tuned,
                           task = ZooTask,
                           resampling = kfold_cv,
                           measures = list(acc, mmce)
)

dtree_tuned_cv$aggr


dtree
dtree_tuned_trained <- train(dtree_tuned, ZooTask)
dtree_tuned_trained_model <- getLearnerModel(dtree_tuned_trained)
rpart.plot(dtree_tuned_trained_model)



# What is boostrapping ----------------------------------------------------

x <- rnorm(10)

# one bootstrap resample
sample(x) # sampling *without* replacement
sample(x, replace = TRUE) # sampling *with* replacement



# Random forest -----------------------------------------------------------

rforest <- makeLearner('classif.randomForest')

# beware of over-fitting!
rforest_trained <- train(rforest, ZooTask)

calculateConfusionMatrix(predict(rforest_trained, newdata = Zoo))
performance(predict(rforest_trained, newdata = Zoo), measures = list(acc, mmce))

# cross-validation

rforest_cv <- resample(learner = rforest,
                       task = ZooTask,
                       resampling = kfold_cv,
                       measures = list(acc, mmce))

rforest_cv$aggr

getParamSet(rforest)

rforest_params_set <- makeParamSet(
  makeIntegerParam('ntree', lower = 250, upper = 500), # number of decision trees
  makeIntegerParam('mtry', lower = 6, upper = 12),     # no. of features to try
  makeIntegerParam('nodesize', lower = 2, upper = 6),  # min no. of cases per node
  makeIntegerParam('maxnodes', lower = 5, upper = 10)  # no. of leaves
)

rand_search <- makeTuneControlRandom(maxit = 250)

cv_for_tuning <- makeResampleDesc(method = 'RepCV', 
                                  folds = 5, 
                                  reps = 2)

rforest_tuned <- tuneParams(rforest,
                            task = ZooTask,
                            resampling = cv_for_tuning,
                            par.set = rforest_params_set,
                            control = rand_search)

rforest_tuned$x

# set the rforest params to their optimal values
tuned_rforest <- setHyperPars(rforest, 
                              par.vals = rforest_tuned$x)

# cross validation of our optimal random forest
tuned_rforest_cv <- resample(learner = tuned_rforest,
                             task = ZooTask,
                             resampling = kfold_cv,
                             measures = list(acc, mmce)
)

tuned_rforest_cv$aggr


tuned_rforest_trained <- train(learner = tuned_rforest, task = ZooTask)
predict(tuned_rforest_trained, newdata = newZoo)



# K means clustering ------------------------------------------------------

blobs_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl02/main/data/blobs3.csv")
blobs_df

ggplot(data = blobs_df,
       aes(x = x, y = y, colour = factor(label))
) + geom_point()

ggplot(data = blobs_df,
       aes(x = x, y = y)
) + geom_point()

