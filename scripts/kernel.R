library(keras)
library(EBImage)

# Load MNIST dataset
mnist <- dataset_mnist()
trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

# Plot images
par(mfrow = c(3,3))
invisible(lapply(1:9, function(i) plot(as.raster(trainx[i,,], max = 255))))
par(mfrow = c(1,1))

# Select five images
a <- c(1, 12, 36, 48, 66, 101, 133, 139, 146)
par(mfrow = c(3,3))
invisible(lapply(a, function(i) plot(as.raster(trainx[i,,], max = 255))))
par(mfrow = c(1,1))

# Reshape and rescale data
trainx <- array_reshape(trainx, c(nrow(trainx), 784)) / 255
testx <- array_reshape(testx, c(nrow(testx), 784)) / 255

# One hot encoding
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

# Define model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units= 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% 
  compile(loss = 'categorical_crossentropy', optimizer = optimizer_rmsprop(), metrics = 'accuracy')

# Fit model
history <- model %>% 
  fit(trainx, trainy, epochs = 30, batch_size = 32, validation_split = 0.2)

# Evaluate and predict on test data
model %>% evaluate(testx, testy)
pred <- model %>% predict(testx) %>% k_argmax()
prob <- model %>% predict(testx)
save_model_hdf5(model, "D:/RStudioLabs/TensorFlow/model/my_MNIST_model.hdf5")

# Load saved model
loaded_model <- load_model_hdf5('D:/RStudioLabs/TensorFlow/model/my_MNIST_model.hdf5')

# Load new data
setwd("D:/RStudioLabs/TensorFlow/tests")
temp <- list.files(pattern = "*.jpg")
mypic <- lapply(temp, function(file) {
  img <- readImage(file)
  colorMode(img) <- Grayscale
  img <- 1 - img
  img <- resize(img, 28, 28)
  img <- array_reshape(img, c(28,28,3))
})

# Predict on new data
newx <- do.call(rbind, lapply(mypic, `[`, 1:784))
newy <- c(7,5,2,0,5,3,4,3,2,7,5,6,8,5,6)
pred <- loaded_model %>% predict(newx) %>% k_argmax()

# Display results
table_values <- table(Predicted = as.vector(pred), Actual = as.vector(newy))

table_values 

(diag(table_values))

