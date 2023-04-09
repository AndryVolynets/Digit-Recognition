print <- function(path){
  
  setwd(path)
  
  temp = list.files(pattern = "*.jpg")
  mypic <- list()
  
  for (i in 1:length(temp)) {
    mypic[[i]] <- readImage(temp[[i]])
  }
  
  for (i in 1:length(temp)) {
    colorMode(mypic[[i]]) <- Grayscale
  }
  
  par(mfrow = c(4,4))
  for (i in 1:length(temp)) 
    plot(mypic[[i]])
}