#############
##LIBRARIES##
#############

library(tidymodels) 
library(tidyverse)
library(vroom) 
library(glmnet)
library(randomForest)
library(doParallel)
library(timetk)

####################
##WORK IN PARALLEL##
####################

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

########
##DATA##
########

my_data <- vroom("train.csv")
test_data <- vroom("test.csv")

###############
##TIME SERIES##
###############

nStores <- max(my_data$store)
nItems <- max(my_data$item)

############
##MODELING##
############

storeItemTrain1 <- my_data %>%
  filter(store==5, item==5)

storeItemTrain2 <- my_data %>%
  filter(store==6, item==12)


storeItemTrain3 <- my_data %>%
  filter(store==3, item==35)

storeItemTrain4 <- my_data %>%
  filter(store==2, item==15)


#######
##EDA##
#######

storeItemTrain %>%
  plot_time_series(date, sales, .interactive=FALSE)

plot1 <- storeItemTrain1 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

plot2 <- storeItemTrain2%>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

plot3 <- storeItemTrain3 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

plot4 <- storeItemTrain4 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

library(patchwork)


EDA_plot <- (plot1 + plot2) / (plot3 + plot4)
ggsave("EDA_plot.png")
