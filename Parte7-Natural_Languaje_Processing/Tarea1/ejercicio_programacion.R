# Ejercicio de programación - R

install.packages(c("readr", "dplyr", "stringr", "tm", "SnowballC", 
                  "caret", "glmnet", "class", "e1071", 
                  "rpart", "randomForest", "tidyr", "ggplot2"))



# Cargamos Librerías

library(readr)
library(dplyr)
library(stringr)
library(tm)
library(SnowballC)
library(caret)
library(glmnet)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(tidyr)
library(ggplot2)

# Importar el conjunto de datos
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)


corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)


# Crear el modelo de Bolsa de Palabras
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked


# Codificar la característica objetivo como factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))


# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Ajustar diferentes modelos a nuestro dataset
# Logistic Regression
logistic_model = glm(formula = Liked ~ .,
                     family = binomial,
                     data = training_set)
logistic_pred = predict(logistic_model, newdata = test_set, type = 'response')
logistic_pred = ifelse(logistic_pred > 0.5, 1, 0)
logistic_pred = factor(logistic_pred, levels = c(0, 1))

# Evaluamos el modelo
conf_log_reg = confusionMatrix(logistic_pred, test_set$Liked)
print(conf_log_reg)


# KNN
knn_model = knn(train = training_set[, -ncol(training_set)], test = test_set[, -ncol(test_set)], cl = training_set$Liked, k = 5)

# Evaluamos el modelo
conf_knn = confusionMatrix(knn_model, test_set$Liked)
print(conf_knn)

# Naive Bayes
naive_bayes_model = naiveBayes(Liked ~ ., data = training_set)
naive_bayes_pred = predict(naive_bayes_model, newdata = test_set)

# Evaluamos el modelo
conf_nb = confusionMatrix(naive_bayes_pred, test_set$Liked)
print(conf_nb)

# Support Vector Machine (SVM)
svm_model = svm(Liked ~ ., data = training_set, type = 'C-classification', kernel = 'linear')
svm_pred = predict(svm_model, newdata = test_set)

# Evaluamos el modelo
conf_svm = confusionMatrix(svm_pred, test_set$Liked)
print(conf_svm)

# Árbol de Decisión
tree_model = rpart(Liked ~ ., data = training_set)
tree_pred = predict(tree_model, newdata = test_set, type = "class")

# Evaluamos el modelo
conf_dt = confusionMatrix(tree_pred, test_set$Liked)
print(conf_dt)


# Random Forest
rf_model = randomForest(x = training_set[-692], y = training_set$Liked, ntree = 500)
rf_pred = predict(rf_model, newdata = test_set[-692])

# Evaluamos el modelo
conf_rf = confusionMatrix(test_set[, 692], rf_pred)
print(conf_rf)




#######################




# Crear una función para extraer métricas
extract_metrics <- function(conf_matrix, model_name) {
  accuracy <- conf_matrix$overall["Accuracy"]
  precision <- conf_matrix$byClass["Pos Pred Value"]
  recall <- conf_matrix$byClass["Sensitivity"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  data.frame(
    Model = model_name,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  )
}


# Evaluar y consolidar métricas de cada modelo
results_table <- rbind(
  extract_metrics(conf_log_reg, "Logistic Regression"),
  extract_metrics(conf_knn, "KNN"),
  extract_metrics(conf_nb, "Naive Bayes"),
  extract_metrics(conf_svm, "Support Vector Machine"),
  extract_metrics(conf_dt, "Decision Tree"),
  extract_metrics(conf_rf, "Random Forest")
)

# Imprimir la tabla consolidada
print(results_table)



######################


install.packages("C50")     # Para el modelo C5.0

# CART
cart_model = rpart(Liked ~ ., data = training_set, method = "class")
cart_pred = predict(cart_model, newdata = test_set, type = "class")

# Evaluamos el modelo
conf_cart = confusionMatrix(cart_pred, test_set$Liked)
print(conf_cart)


# C5.0
c50_model = C5.0(Liked ~ ., data = training_set)
c50_pred = predict(c50_model, newdata = test_set)

# Evaluamos el modelo
conf_c50 = confusionMatrix(c50_pred, test_set$Liked)
print(conf_c50)


# Máxima Entropía con glmnet
library(glmnet)

# Convertir los datos a matrices
train_matrix = as.matrix(training_set[, -ncol(training_set)])
test_matrix = as.matrix(test_set[, -ncol(test_set)])

# Entrenar el modelo de Máxima Entropía
maxent_model = cv.glmnet(train_matrix, as.factor(training_set$Liked), family = "multinomial", type.measure = "class")

# Realizar predicciones
maxent_pred = predict(maxent_model, newx = test_matrix, s = "lambda.min", type = "class")
maxent_pred = as.factor(maxent_pred)

# Evaluar el modelo
conf_maxent = confusionMatrix(maxent_pred, test_set$Liked)
print(conf_maxent)







# Evaluar y consolidar métricas de cada modelo
results_table <- rbind(
  extract_metrics(conf_log_reg, "Logistic Regression"),
  extract_metrics(conf_knn, "KNN"),
  extract_metrics(conf_nb, "Naive Bayes"),
  extract_metrics(conf_svm, "Support Vector Machine"),
  extract_metrics(conf_dt, "Decision Tree"),
  extract_metrics(conf_rf, "Random Forest"),
  extract_metrics(conf_cart, "CART"),
  extract_metrics(conf_c50, "C5.0"),
  extract_metrics(conf_maxent, "Maxent")
)

# Imprimir la tabla consolidada
print(results_table)

