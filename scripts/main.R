library(keras)
library(EBImage)
source("../scripts/PrintPictures.R")

# Load MNIST dataset
mnist <- dataset_mnist()
trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

# Reshape and rescale data
trainx <- array_reshape(trainx, c(nrow(trainx), 28, 28, 1)) / 255
testx <- array_reshape(testx, c(nrow(testx), 28, 28, 1)) / 255

# One hot encoding
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

# Define model
model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% 
  compile(loss = 'categorical_crossentropy', optimizer = optimizer_rmsprop(), metrics = 'accuracy')

# Fit model
history <- model %>% 
  fit(trainx, trainy, epochs = 50, batch_size = 32, validation_split = 0.2)

# Evaluate and predict on test data
model %>% evaluate(testx, testy)
pred <- model %>% predict(testx) %>% k_argmax()
prob <- model %>% predict(testx)

save_model_hdf5(model, "../model/my_MNIST_model.hdf5")

# Load saved model
loaded_model <- load_model_hdf5('../model/my_MNIST_model.hdf5')

# Load new data
temp <- list.files(path = "../tests", pattern = "*.jpg")
mypic <- lapply(temp, function(file) {
  img <- readImage(file)
  colorMode(img) <- Grayscale
  img <- 1 - img
  img <- resize(img, 28, 28)
  img <- array_reshape(img, c(28,28,3))
})

# Predict on new data
newx <- array_reshape(newx, c(15, 28, 28, 1))
newy <- c(7,5,2,0,5,3,4,3,2,7,5,6,8,5,6)
pred <- loaded_model %>% predict(newx) %>% k_argmax()

# Display results
table_values <- table(Predicted = as.vector(pred), Actual = as.vector(newy))
table_values 
(diag(table_values))


getwd()
