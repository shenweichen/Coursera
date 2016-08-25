#Download Link:https://d396qusza40orc.cloudfront.net/rprog%2Fdata%2Fspecdata.zip
corr <- function(directory,threshold =0,id =1:332 ){
    filename <-as.character(id)
    for (i in 1:length(filename)) {
        filename[i]<- paste(filename[i],"csv",sep = ".")
        
        if(nchar(filename[i])==5){
            filename[i] <- paste("00",filename[i],sep = "")
        }
        else if(nchar(filename[i])==6){
            filename[i] <- paste("0",filename[i],sep = "")
        }
    }
    
    cordata <- vector("numeric")
    count <- 0L
    for(i in 1:length(id)){
        datapath <- paste(getwd(),directory,filename[i],sep = "/")
        data <- read.csv(datapath)
        temp<-(!is.na(data$sulfate))&(!is.na(data$nitrate))
        if(sum(temp)>threshold){
            count <- count +1
            cordata[count] <- cor(data$sulfate,data$nitrate,"complete.obs")
        }
    }
    cordata
}