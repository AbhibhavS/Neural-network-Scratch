
set.seed(123)
data<-as.matrix(iris) #iris dataset

classes<-unique(data[,5])# retreive unique classes


###########ONE HOT ENCODING###########
#Initializing an empty matrix
hot_encode<-matrix(0, ncol=length(unique(classes)), byrow = T,
                   nrow= nrow(data), 
                   dimnames = list(NULL, c(unique(classes))))
#Imputing 1 at respective class
for(i in 1:nrow(data)){
  for(j in 1:3){
    if(data[i,5] == classes[j]){hot_encode[i,j]<-1} 
    else next
  }
}
# Combining the data and encoded labels
data<-as.matrix(cbind(iris[,1:4], hot_encode))
data<-data[sample(1:nrow(data)), ]

set.seed(123)
seq<-sample(1:nrow(data))


head(data)

#########################################################

############ Sigmoid Activation Function######
activation<-function(x){ 
  y<- 1/(1+exp(-x)) #Sigmaod function
  return(y)
}
###########Derivative of Sigmoid Function#####
derivative_activation<-function(x){
  y<-x*(1-x) #derivative of sigmoid function
  return(y)
}
###########Normalization Function############
normalize<-function(x){
  for(i in 1:ncol(x)){
    x[,i]<-(max(x[,i])-x[,i])/(max(x[,i])-min(x[,i]))
  }
  return(x)
}
#Normalizing the input before feeding to an NN is a good practice###
data[, 1:4]<-normalize(data[, 1:4])


#####################################################

neuron_input<- 4 #Define number of neurons in Input layer
neuron_layer1<- 4 #Define number of neurons in first hidden layer
neuron_layer2<- 3 #Define number of neurons in second hidden layer
neuron_output<- 3 #Define number of neurons in the output layer
#initalizing weight W1 and bias b1
set_weight_1<-matrix(runif(4*4, 0,1), 
                     ncol= neuron_layer1, nrow= neuron_input)
bias_1<-runif(neuron_layer1, 0,1)
#initalizing weight W2 and bias b2
set_weight_2<-matrix(runif(4*3, 0,1), 
                     ncol= neuron_layer2, nrow= neuron_layer1)
bias_2<-runif(neuron_layer2, 0,1)
#initalizing weight W3 and bias b3
set_weight_3<-matrix(runif(3*3, 0,1), 
                     ncol= neuron_output, nrow= neuron_layer2)
bias_3<-runif(neuron_output, 0,1)
################# TRAINING SET #################
input_layer<-data[1:120, 1:4]
label<-data[1:120, 5:7]
################# TEST SET #####################
test<-data[121:150, 1:4]
test_label<-as.integer(iris$Species[seq[121:150]])
#--------------------------------------------------------#
lr=0.1 # Learning Rate
er<-NA  # The performance function value
itr<-1  # Iteration
accuracy<-NA #Training Accuracy
t.accuracy<-NA #Test Accuracy
loss<-NA #loss vector containing the error value at current epoch

#-----------------------------------------------------#

while(itr <= 5000){
  
  
  print(paste("epoch =", itr)) #print the current epoch
  itr<-itr+1 #Update the iteration number
  ###############FORWARD FEED##################################
  #-----------------------STEP 1-----------------------------#
  hidden_layer_1<-t(t(input_layer %*% set_weight_1) + bias_1) 
  activated_hidden_layer_1<-activation(hidden_layer_1)
  
  #-----------------------STEP 2-----------------------------#
  hidden_layer_2<-t(t(activated_hidden_layer_1 %*% set_weight_2) + bias_2)
  activated_hidden_layer_2<-activation(hidden_layer_2)
  #-----------------------STEP3------------------------------#
  final_layer<-activation(t(t(activated_hidden_layer_2 %*% set_weight_3) + bias_3))
  #-----------------------STEP4------------------------------#
  er<-sum(((label-final_layer)^2)/2)/120 
  error<- -(label-final_layer)
  loss[itr]<-er
  ###################BACKPROPOGATION#################################
  #-------------------------STEP5-----------------------------#
  derivation_final_layer<-derivative_activation(final_layer)
  delta_final_layer<- derivation_final_layer * error
  #-------------------------STEP6-----------------------------#
  derivative_hidden_layer_2<-derivative_activation(activated_hidden_layer_2)
  error_layer_2<-delta_final_layer%*%t(set_weight_3)
  delta_layer_2<- derivative_hidden_layer_2 * error_layer_2
  #-------------------------STEP7------------------------------#
  derivative_hidden_layer_1<-derivative_activation(activated_hidden_layer_1) 
  error_layer_1<- delta_layer_2 %*% t(set_weight_2)
  delta_layer_1<- derivative_hidden_layer_1 * error_layer_1
  
  #####################UPDATE##################################
  #-------------------------STEP8-----------------------------#
  set_weight_3 <-set_weight_3 -
    lr*t(activated_hidden_layer_2)%*%delta_final_layer
  
  #---------------------------STEP9--------------------------#
  set_weight_2 <-set_weight_2 -
    lr*t(activated_hidden_layer_1)%*%delta_layer_2
  
  #--------------------------STEP10--------------------------#
  set_weight_1 <-set_weight_1 -
    lr*t(input_layer)%*%delta_layer_1
  
  #--------------------------STEP11--------------------------#
  bias_3 <- bias_3 - lr* colSums(delta_final_layer)
  bias_2 <- bias_2 - lr* colSums(delta_layer_2)
  bias_1 <- bias_1 - lr* colSums(delta_layer_1)
  
  ######################################################
  
  prediction<-NA
  for(i in 1:nrow(final_layer)){
    prediction[i]<-(which(final_layer[i,]== max(final_layer[i,])))
  }
  actual<-as.integer(iris$Species[seq[1:120]])
  
  result<-table(prediction, actual)
  accuracy[itr]<- sum(diag(result))/sum(result)
  
}


################Prediction function###########

predict<-function(test, label){
  
  t.hidden_layer_1<-t(t(test %*% set_weight_1) + bias_1)
  t.activated_hidden_layer_1<-activation(t.hidden_layer_1)
  
  t.hidden_layer_2<-t(t(t.activated_hidden_layer_1 %*% set_weight_2) + bias_2)
  t.activated_hidden_layer_2<-activation(t.hidden_layer_2)
  
  t.final_layer<-activation(t(t(t.activated_hidden_layer_2 %*% set_weight_3) + bias_3))
  t.prediction<-NA
  for(i in 1:nrow(t.final_layer)){
    t.prediction[i]<-(which(t.final_layer[i,]== max(t.final_layer[i,])))
  }
  t.actual<-label
  t.result<-table(t.prediction, t.actual)
  colnames(t.result)<-unique(iris$Species)
  row.names(t.result)<-unique(iris$Species)
  t.accuracy<- sum(diag(t.result))/sum(t.result)
  result<-list(t.result, t.accuracy)
  names(result)<- c("Confusion Matrix", "Result")
  
  return(result)
  
}

predict(test,test_label)



############### PLOT ACCURACY and LOSS###################
par(bg="black")
plot(NA, type = "l", col="red",
     xlim=c(0,100), ylim=c(0,1), lwd=4, 
     xlab="Epochs", ylab="Value", xaxt="n", yaxt="n")
abline(h=seq(0, 1, 0.1), col="white", lwd=0.1)
par(new=T)
plot(accuracy[seq(1, 5000, 50)], type = "l", col="red",
     xlim=c(0,100), ylim=c(0,1), lwd=6, 
     xlab="Epochs", ylab="Value", xaxt="n", yaxt="n")
par(new=T)
plot(loss[seq(1, 5000, 50)], type = "l", col="blue",
     xlim=c(0,100), ylim=c(0,1), lwd=6,
     xlab="Epochs", ylab="Value", xaxt="n", yaxt="n")
axis(1, at=seq(0, 100, 2), labels = seq(0, 5000, 100), 
     font=9, cex=1, col.axis="white", col.ticks = "White")
axis(2, at=seq(0, 1, 0.1), labels = seq(0, 1, 0.1), 
     font=9, cex=1, col.axis="white", col.ticks = "White")

legend("right", lwd=4, col=c("blue", "red"), bty="n",
       legend = c("Loss", "Accuracy"), text.font=9, cex=1, text.col="white")
box(lwd=2, col="white")


