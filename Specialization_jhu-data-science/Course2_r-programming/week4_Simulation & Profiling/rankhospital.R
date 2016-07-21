#Guide:https://d3c33hcgiwev3.cloudfront.net/_775189147d7b89d66333adf6d920b52d_ProgAssignment3v2.pdf?Expires=1469232000&Signature=F0HqsOde4QpGSr1-u91-5aisJD3~T1D3LKlBC5B39PqmcdkK9UrcesrsCZ8wwLMHj8DgX7gRl8AKu6bRfWGMkRq8oyvzutDErMaKmeD9yrs93mZ~pEl0rIdUHn5M0Mjk0XF4Aofp9eY1iwaZKXgABCdHj5OMEedk2TnrpycutxM_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A
#Download Link:https://d396qusza40orc.cloudfront.net/rprog%2Fdata%2FProgAssignment3-data.zip
rankhospital <- function(state,outcome,num="best"){
    data <-read.csv("outcome-of-care-measures.csv")
    statename <- as.character(data$State)
    for(i in 1:length(statename)){
        if (statename[i]==state)
            break
        if(i==length(statename))
            stop("invalid state")
    }
    
    data <- subset.data.frame(data,State==state)
    if(outcome=="heart attack"){
        data <- data.frame(name=data$Hospital.Name,rate=data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack)
    }else if(outcome=="heart failure"){
        data <- data.frame(name=data$Hospital.Name,rate=data$Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure)
    }else if(outcome=="pneumonia"){
        data <- data.frame(name=data$Hospital.Name,rate=data$Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia)
    }else{
        stop("invalid outcome")
    }

    if(num=="best")
        num<-1
    else if(num=="worst")
        num<-nrow(data)
    
    hospitalname<-data$name[order(data$rate,data$name)[num]]
    as.character(hospitalname)

}