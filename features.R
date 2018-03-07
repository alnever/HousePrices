library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)

imputeNumeric <- function(col) {
  x.mean <- mean(col, na.rm = TRUE)
  col[is.na(col)] <- x.mean
  return(col)
}

imputeFactors <- function(col) {
  levels(col) <- c(levels(col), "Missing")
  col[is.na(col)] <-"Missing"
  return(col)
}

imputeDataSet <- function(ds) {
  tmp <- ds
  for (col in names(tmp)) {
    if (sum(is.na(tmp[[col]])) > 0) {
      if (is.factor(tmp[[col]])) {
        tmp[[col]] = imputeFactors(tmp[[col]])
      } 
      else if (is.numeric(tmp[[col]])) {
        tmp[[col]] = imputeNumeric(tmp[[col]])
      }
    }
  }
  return(tmp)
}


checkNas <- function(ds) {
  columns <- names(ds)
  col.nas <- sapply(columns, function(c) {sum(is.na(ds[[c]]))})
  pcn.nas <- sapply(columns, function(c) {sum(is.na(ds[[c]])) / length(ds[[c]])})
  num.nas <- data.frame(Column = columns, NumNas = col.nas, PcnNas = pcn.nas)
  return(num.nas)
}

removeHighLevelNas <- function(data, nas.info, nas.level = .9){
  col.stay <- as.character(nas.info[nas.info$PcnNas < nas.level, 'Column'])
  return(data[, col.stay])
}

FactorToInt <- function(col) {
  lev <- levels(col)
  y <- sapply(col, function(x) {which(lev == x)} )
  return(y)
}

factorizeDS <- function(ds) {
  tmp = ds
  for (col in names(tmp)) {
    if (is.factor(tmp[[col]]))
      tmp[[col]] = FactorToInt(tmp[[col]])
  }
  return(tmp)
}

## Read data and select Id value
data <- read.csv("train.csv")
data.id <- data$Id
data <- data %>% select(-Id)

## Handling NAs values
data.nas <- checkNas(data)
data <- removeHighLevelNas(data, data.nas)
data <- imputeDataSet(data)
checkNas(data)

## Covert factors to Numeric
data <- factorizeDS(data)
sum(checkNas(data)$NumNas >0)

## Handling NZV
data.SalePrice <- data$SalePrice
data.nzv <- nearZeroVar(data[,-dim(data)[2]], saveMetrics = TRUE, names = TRUE)
data.nzv
nzv.vars <- data.nzv[data.nzv$zeroVar == FALSE  &  data.nzv$nzv == FALSE,]
data <- data[, rownames(nzv.vars)]
data['SalePrice'] =  data.SalePrice

##Exploring correlation of variables
data.cor = round(cor(data), 2)
data.mlt = melt(data.cor)
good.cor  = data.mlt %>%
  filter(Var1 == "SalePrice" & value > .5) 
data.mlt<- data.mlt %>%
  filter(Var2 %in% good.cor$Var2) %>%
  filter(Var1 %in% good.cor$Var2) 

### Select just variables having high correlation to the targer variable
## data <- data[,good.cor$Var2]


