# Libraries required
library(reticulate)
use_condaenv("keras3env", required = TRUE)
#library(keras)
library(keras3)
library(tensorflow)
#install_tensorflow(method = "conda", envname = "myenv")
#imdb data
imdb <- dataset_imdb(num_words = 500) 
c(c(train_x, train_y), c(test_x, test_y)) %<-% imdb
length(train_x); length(test_x)

table(train_y)
table(test_y)
train_x[[3]]

z <- NULL
for (i in 1:25000) {z[i] <- print(length(train_x[[i]]))}
summary(z)



for (i in 1:10) print(length(train_x[[i]]))


# Padding sequences
train_x <- pad_sequences(train_x, maxlen = 200) 
test_x <- pad_sequences(test_x, maxlen = 200)

# Define the model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 500, output_dim = 32) %>%
  layer_simple_rnn(units = 32,
                   return_sequences = TRUE,
                   activation = 'relu') %>%
  layer_simple_rnn(units = 32,
                   return_sequences = TRUE,
                   activation = 'relu') %>%
  layer_simple_rnn(units = 32,
                   activation = 'relu') %>%
  layer_dense(units = 1, activation = "sigmoid")
model
# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = "adam",
  metrics = c("acc")
)

# Train the model (make sure train_x and train_y are defined)
model %>% fit(train_x, train_y, epochs = 10, batch_size = 32, 
              validation_split = 0.2)

# Fit model
history <- model %>% fit(train_x, train_y,
                         epochs = 25,
                         batch_size = 128,
                         validation_split = 0.2)
plot(history)

# Prediction
model %>% evaluate(train_x, train_y)
pred <- model %>%   predict(train_x)
pred <- ifelse(pred <0.5, 0, 1)
table(Predicted=pred, Actual=imdb$train$y)

model %>% evaluate(test_x, test_y)
pred1 <- model %>%   predict(test_x)
pred1 <- ifelse(pred1 <0.5, 0, 1)
table(Predicted=pred1, Actual=imdb$test$y)
