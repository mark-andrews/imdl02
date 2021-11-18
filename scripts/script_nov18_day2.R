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
