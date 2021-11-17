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

