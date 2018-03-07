library(xgboost)

source("training_mlr.R")

x_train <- p_train
y_train <- getTaskTargets(tsk.train)
train <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)

x_valid <- p_valid
y_valid <- getTaskTargets(tsk.valid)
valid <- xgb.DMatrix(data = as.matrix(x_valid), label = y_valid)

watchlist <- list(eval = valid, train = train)
param <- list(
  booster = "gblinear",
  objective = "reg:linear",
  eval_metric = "rmse",
  seed = 0
)


xgb.model <- xgb.train(params = param, 
                       data = train, 
                       nrounds = 20000, 
                       watchlist = watchlist,
                       print_every_n = 50)

xgb.predict <- predict(xgb.model, newdata = as.matrix(x_valid))

