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


# create task and learner -------------------------------------------------

blobsTask <- makeClusterTask(
  data = as.data.frame(dplyr::select(blobs_df, -label)),
)

k_means <- makeLearner('cluster.kmeans')

getParamSet(k_means)

k_means_param_set <- makeParamSet(
  makeDiscreteParam('centers', values = 1:8)
)

grid_search <- makeTuneControlGrid()

kfold_cv <- makeResampleDesc('RepCV', folds = 10, reps = 10)

k_means_tuning <- tuneParams(learner = k_means,
                             task = blobsTask,
                             resampling = kfold_cv,
                             control = grid_search,
                             par.set = k_means_param_set)
           
k_means_tuning$x
k_means_tuned <- setHyperPars(k_means, par.vals = k_means_tuning$x)
           

k_means_tuned_trained <- train(k_means_tuned, blobsTask)
k_means_model <- getLearnerModel(k_means_tuned_trained)
k_means_model$centers
k_means_model$cluster

blobs_df %>% 
  mutate(cluster = factor(k_means_model$cluster)) %>% 
  ggplot(aes(x = x, y = y, colour = cluster)) +
  geom_point()

table(blobs_df$label, k_means_model$cluster)


blobs_df %>% 
  ggplot(aes(x = x, y = y, colour = factor(label))) +
  geom_point()



# Mixture models ----------------------------------------------------------

ggplot(faithful,
       aes(x = eruptions, y = waiting)
) + geom_point() + geom_density2d()


# load mclust
library(mclust)

# evaluate K, the number of components
faithful_bic <- mclustBIC(faithful)
plot(faithful_bic)

M <- Mclust(faithful, x = faithful_bic)
summary(M, parameters = TRUE)

plot(M, what = 'classification')
plot(M, what = 'density')




# Neural nets -------------------------------------------------------------

library(torch)

gvhd_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl02/main/data/gvhd.csv")

cuda_is_available()

X <- as.matrix(select(gvhd_df, starts_with('CD')))
X <- torch_tensor(X, dtype = torch_float())

y <- torch_tensor(gvhd_df$type, dtype = torch_long())

mlp <- nn_sequential(
  
  # input to hidden mapping
  nn_linear(4, 2),
  nn_relu(),
  
  # hidden to output mapping
  nn_linear(2, 2),
  nn_softmax(2)
  
)

mlp(X)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(mlp$parameters, lr = 0.01)

iters <- 500
N <- nrow(gvhd_df)
for (i in 1:iters){
  
  optimizer$zero_grad()
  
  y_pred <- mlp(X)
  
  loss <- criterion(y_pred, y) # evaluating model performance
  
  # backprop 
  loss$backward()
  
  # make a step in param space
  optimizer$step()

  accuracy <- as_array(sum(y == y_pred$argmax(dim = 2) )/N)
  
  print(accuracy)

}



# Looking more at this MLP  -----------------------------------------------

library(torch)
library(tidyverse)

gvhd_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl02/main/data/gvhd.csv")

train_test_split <- function(N, split = 0.8){
  train_idx <- sample(seq(N), size = split * N)
  test_idx <- setdiff(seq(N), train_idx)
  list(train_idx = train_idx,
       test_idx = test_idx)
}

tt_split <- train_test_split(nrow(gvhd_df))

# 2. Convert our input data to matrices and labels to vectors.
x_train <- as.matrix(gvhd_df[tt_split$train_idx,] %>% select(starts_with('CD')))
y_train <- as.numeric(gvhd_df[tt_split$train_idx,] %>% pull(type))
x_test <- as.matrix(gvhd_df[tt_split$test_idx,] %>% select(starts_with('CD')))
y_test <- as.numeric(gvhd_df[tt_split$test_idx,] %>% pull(type))

# 3. Convert our input data and labels into tensors.
x_train <- torch_tensor(x_train, dtype = torch_float())
y_train <- torch_tensor(y_train, dtype = torch_long())
x_test <- torch_tensor(x_test, dtype = torch_float())
y_test <- torch_tensor(y_test, dtype = torch_long())

# Make a multilayer perceptron
mlp <- nn_sequential(
  
  # Layer 1
  nn_linear(4, 8),
  nn_relu(), 
  
  # Layer 2
  nn_linear(8, 2),
  nn_softmax(2)
  
)



# test that it works (forward pass; prediction)
pred_test <- mlp(x_train)

# create an objective function
criterion <- nn_cross_entropy_loss()  

# calculate objective function
criterion(pred_test, y_train)

# Define optimizer
optimizer = optim_adam(mlp$parameters, lr = 0.01)

epochs = 500

# Train the network
for(i in 1:epochs){
  
  optimizer$zero_grad()
  
  y_pred = mlp(x_train)
  loss = criterion(y_pred, y_train)
  
  loss$backward()
  optimizer$step()
  
  
  # Check Training
  if(i %% 10 == 0){
    
    accuracy = (y_train == y_pred$argmax(dim=2))$sum()$item() / y_train$size()
    
    cat("iter:", i, "Loss: ", loss$item()," Accuracy:",accuracy,"\n")
  }
  
}


# prediction --------------------------------------------------------------

mlp_pred <- mlp(x_test)$argmax(dim=2) %>% as_array() %>% as_factor()
mlp_truth <- y_test %>% as_array() %>% as_factor()

caret::confusionMatrix(mlp_pred, mlp_truth)$table


# Logistic regression comparison ------------------------------------------

library(mlr)
gvhd_df$type <- as.factor(gvhd_df$type)

gvhdTask <- makeClassifTask(data = gvhd_df[tt_split$train_idx,], target = 'type')
logReg <- makeLearner('classif.logreg', predict.type = 'prob')
gvhd_trained <- train(logReg, gvhdTask)
p <- predict(gvhd_trained, newdata = gvhd_df[tt_split$test_idx,])
calculateConfusionMatrix(p)



# Convolution neural network ----------------------------------------------

library(torch)
library(torchvision)

# Download mnist (handwritten digits) training and test sets
# these will download to your working directory
train_ds <- mnist_dataset(
  ".",
  download = TRUE,
  train = TRUE,
  transform = transform_to_tensor
)

test_ds <- mnist_dataset(
  ".",
  download = TRUE,
  train = FALSE,
  transform = transform_to_tensor
)


# but the data into batches of size 32 (32 images)
# this is for computational efficiency
train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 32)

# Make the conv net -------------------------------------------------------

net <- nn_module(
  
  "mnist_convnet",
  
  # create the various layers and functions
  # where do numbers like 9216 come from?
  # Play with functions like in `forward`
  # to see how the tesors are transformed in size by these operations
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      self$fc2()
  }
)

model <- net() # create an instance of the above conv net
optimizer <- optim_adam(model$parameters)

# Train it (this will take time ~ 20-30 mins on a high end cpu)
for (epoch in 1:5) {
  
  l <- c()
  
  for (b in enumerate(train_dl)) {
    # make sure each batch's gradient updates are calculated from a fresh start
    optimizer$zero_grad()
    # get model predictions
    output <- model(b[[1]])
    # calculate loss
    loss <- nnf_cross_entropy(output, b[[2]])
    # calculate gradient
    loss$backward()
    # apply weight updates
    optimizer$step()
    # track losses
    l <- c(l, loss$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}

