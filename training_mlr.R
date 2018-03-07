library(mlr)
library(Metrics)
library(mlrHyperopt)

#Get data from prepared dataset
source("features.R")
set.seed(0)
### Convert all integer values to the numeric
for (col in names(data)){
  data[[col]] = as.numeric(data[[col]])
}

## Data normalization
data.norm <- normalizeFeatures(data, 'SalePrice')

#Prepare data to the training
data.log <- data.norm
data.log$SalePrice <- log(data.norm$SalePrice + 1)

#Create mlr task
tsk = makeRegrTask(data = data.log, target = 'SalePrice')

# Split task into training and validating tasks
ho  = makeResampleInstance("Holdout", tsk, split = .7)
tsk.train = subsetTask(tsk, ho$train.inds[[1]])
tsk.valid = subsetTask(tsk, ho$test.inds[[1]])

# Tune learners
# 
# lrns = c("regr.extraTrees", "regr.randomForest")
# lrns = makeLearners(lrns)
# tsk = tsk.train
# rr = makeResampleDesc('CV', stratify = TRUE, iters = 10)
# lrns.tuned = lapply(lrns, function(lrn) {
#   if (getLearnerName(lrn) == "xgboost" ) {
#     # for xgboost we download a custom ParConfig from the Database
#     pcs = downloadParConfigs(learner.name = getLearnerName(lrn))
#     pc = pcs[[1]]
#   } else {
#     pc = getDefaultParConfig(learner = lrn)
#   }
#   ps = getParConfigParSet(pc)
#   # some parameters are dependend on the data (eg. the number of columns)
#   ps = evaluateParamExpressions(ps, dict = mlrHyperopt::getTaskDictionary(task = tsk))
#   lrn = setHyperPars(lrn, par.vals = getParConfigParVals(pc))
#   ctrl = makeTuneControlRandom(maxit = 20)
#   makeTuneWrapper(learner = lrn, resampling = rr, par.set = ps, control = ctrl)
# })
# res = benchmark(learners = c(lrns, lrns.tuned), tasks = tsk, resamplings = cv10)
# plotBMRBoxplots(res) 

#Exprore different learners
lrn.list = c("regr.extraTrees", "regr.randomForest")
lrn.res = NULL
p_train = NULL
p_test = NULL

cv      = makeResampleDesc("CV", iters = 5)

for (learner in lrn.list) {
  lrn = makeLearner(learner)
  
  
  ## Tuning
  pc = getDefaultParConfig(learner = lrn)
  ps = getParConfigParSet(pc)
  ps = evaluateParamExpressions(ps, dict = mlrHyperopt::getTaskDictionary(task = tsk.train))
  lrn = setHyperPars(lrn, par.vals = getParConfigParVals(pc))
  ctrl = makeTuneControlRandom(maxit = 5)
  lrn = makeTuneWrapper(learner = lrn, resampling = cv, par.set = ps, control = ctrl)
  
  # 
  res     = resample(lrn, tsk.train, cv, list(mlr::rmse, mlr::mse))
  model = mlr::train(lrn, tsk.train)
  valid.p  = predict(model,tsk.valid)
  train.p  = predict(model,tsk.train)
  x <- performance(valid.p, list(mlr::rmse, mlr::mse))
  
  d = data.frame(Learner = learner, mse = x[2], rmse = x[1])
  if (is.null(lrn.res)) {
    lrn.res = d
  } else {
    lrn.res = rbind(lrn.res, d)
  }
  
  if (is.null(p_train)) {
    p_train = data.frame(learner = train.p$data$response)
    p_valid = data.frame(learner = valid.p$data$response)
    names(p_train) = c(learner)
    names(p_valid) = c(learner)
    
  } else {
    p_train = cbind(p_train, data.frame(learner = train.p$data$response))
    p_valid = cbind(p_valid, data.frame(learner = valid.p$data$response))
    append(names(p_train), learner)
    append(names(p_valid), learner)
  }
}


