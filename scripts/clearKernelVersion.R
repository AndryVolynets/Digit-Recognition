library(keras)

mnist <- dataset_mnist()
trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

# Plot images
par(mfrow = c(3,3))
for (i in 1:9) 
  plot(as.raster(trainx[i,,], max = 255))
par(mfrow= c(1,1))

# Five 
a <- c(1, 12, 36, 48, 66, 101, 133, 139, 146)
par(mfrow = c(3,3))

for (i in a) 
  plot(as.raster(trainx[i,,], max = 255))
par(mfrow= c(1,1))

# Reshape & rescale
trainx <- array_reshape(trainx, c(nrow(trainx), 784))
testx <- array_reshape(testx, c(nrow(testx), 784))
trainx <- trainx / 255
testx <- testx /255

# One hot encoding
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

# Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units= 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax')

# Compile
model %>% 
  compile(loss = 'categorical_crossentropy', optimizer = optimizer_rmsprop(), metrics = 'accuracy')

# Fit model
history <- model %>% 
  fit(trainx, trainy, epochs = 30, batch_size = 32, validation_split = 0.2)

# Evaluation and Prediction - Test data
model %>% evaluate(testx, testy)

pred <- model %>% predict(testx) %>% k_argmax()

prob <- model %>% predict(testx)

save_model_hdf5(model, "D:/RStudioLabs/TensorFlow/model/my_MNIST_model.hdf5")


# New data
library(EBImage)

#install.packages("BiocManager")
#BiocManager::install("EBImage")

loaded_model = load_model_hdf5('D:/RStudioLabs/TensorFlow/model/my_MNIST_model.hdf5')

setwd("D:/RStudioLabs/TensorFlow/tests")

temp = list.files(pattern = "*.jpg")
mypic <- list()

for (i in 1:length(temp)) {
  mypic[[i]] <- readImage(temp[[i]])
}

par(mfrow = c(4,4))
for (i in 1:length(temp)) 
  plot(mypic[[i]])

for (i in 1:length(temp)) {
  colorMode(mypic[[i]]) <- Grayscale
}

for (i in 1:length(temp)) {
  mypic[[i]] <- 1-mypic[[i]]
}

for (i in 1:length(temp)) {
  mypic[[i]] <- resize(mypic[[i]], 28, 28)
}

str(mypic)
par(mfrow = c(4,5))

for (i in 1:length(temp)) 
  plot(mypic[[i]])
for (i in 1:length(temp)) {
  mypic[[i]] <- array_reshape(mypic[[i]], c(28,28,3))
  }
new <- NULL

for (i in 1:length(temp)) {
  new <- rbind(new, mypic[[i]])
}
newx <- new[,1:784]
# (7,5,2,0,5,3)
newy <- c(7,5,2,0,5,3,4,3,2,7,5,6,8,5,6)

# Prediction
pred <- loaded_model %>% 
          predict(newx) %>% 
          k_argmax()
pred
table(Predicted = as.vector(pred), Actual = as.vector(newy))