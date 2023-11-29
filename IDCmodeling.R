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
library(embed)
library(modeltime)

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

##########
##RECIPE##
##########

storeItem2 <- my_data %>% #Data for this first run
  filter(store==6, item==12)

my_recipe <- recipe(sales~., data=storeItem2)  %>%
                step_date(date, features="doy") %>%
                step_range(date_doy, min=0, max=pi) %>%
                step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
                step_date(date, features = "dow") %>% 
                step_date(date, features = "month") %>% 
                step_date(date, features = "year") %>% 
                # step_holiday(date, holidays = timeDate::listHolidays()) %>% 
                step_mutate(date_weekend = ifelse(date_dow %in% c("Sun","Sat"), 1, 0)) %>% 
                step_lencode_mixed(all_nominal_predictors(), outcome = vars(sales)) %>% 
                # step_lag(date,lag = 365) %>% 
                step_naomit(all_predictors())

prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)


########
##BART##
########

bart_model <- bart(trees = 100,
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune(),
                   prior_outcome_range = tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("regression")

bart_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(bart_model) 

tuning_grid_bart <- grid_regular(prior_terminal_node_coef(),
                                 prior_terminal_node_expo(),
                                 prior_outcome_range(),
                                 levels = 3)
folds_bart <- vfold_cv(storeItem2, v = 5, repeats=1)

CV_results_bart <- bart_workflow %>%
            tune_grid(resamples=folds_bart,
            grid=tuning_grid_bart,
            metrics=metric_set(smape))

bestTune_bart <- CV_results_bart %>%
  select_best("smape")

collect_metrics <- (CV_results) %>% 
  filter(bestTune_bart) %>% 
  pull(mean)



final_bart_wf <- bart_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = my_data)


bart_predictions<- final_bart_wf %>% 
  predict(new_data = test_data)

bart_predictions <- final_bart_wf %>% 
  predict(new_data = test_data, type="class")


bart_predictions <- bind_cols(test_data$id,bart_predictions$.pred_class)

colnames(bart_predictions) <- c("id","type")

bart_predictions <- as.data.frame(bart_predictions)

vroom_write(bart_predictions,"bart_predictions.csv",',')


######
##RF##
######


RF_model <- rand_forest(mode = "regression",
                        mtry = tune(),
                        trees = 500,
                        min_n = tune()) %>% #Applies Linear Model
  set_engine("randomForest")

RF_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_model) 

tuning_grid_rf <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = 3)
folds_rf <- vfold_cv(storeItem2, v = 5, repeats=1)

CV_results_rf <- RF_workflow %>%
  tune_grid(resamples=folds_rf,
            grid=tuning_grid_rf,
            metrics=metric_set(smape))
bestTune_rf <- CV_results_rf %>%
  select_best("smape")

collect_metrics <- (bestTune_rf) %>% 
  filter(.metrics == "smape") %>% 
  slice(1) %>% 
  pull(mean)
class(storeItem2$sales)


#########################
##EXPONENTIAL SMOOTHING##
#########################

train <- my_data %>% filter(store==8, item==20)


cv_split <- time_series_split(train, assess="3 months", cumulative = TRUE)

cv_split %>%
tk_time_series_cv_plan() %>% #Put into a data frame7
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)



ES_model <- exp_smoothing() %>%
set_engine("ets") %>%
fit(sales~date, data=training(cv_split))

cv_results <- modeltime_calibrate(ES_model,
                                  new_data = testing(cv_split))

## Visualize CV results
cv_results %>%
              modeltime_forecast(new_data = testing(cv_split),
              actual_data = train) %>%
                   plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
    modeltime_accuracy() %>%
      table_modeltime_accuracy( .interactive = FALSE)

es_fullfit <- cv_results %>%
  modeltime_refit(data = train)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

  es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plot1 <- cv_results %>%
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plot2 <-es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plot3 <- cv_results %>%
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plot4 <-es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(plot1,plot3,plot2,plot4, nrows = 2)



#########
##ARIMA##
#########

cv_split <- time_series_split(bake_1, assess="3 months", cumulative = TRUE)

cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame7
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2) %>%
  set_engine("auto_arima") %>%
  fit(sales~date, data=training(cv_split))

arima_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

cv_results <- modeltime_calibrate(arima_model,
                                  new_data = testing(cv_split))

cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy( .interactive = FALSE)

arima_fullfit <- cv_results %>%
  modeltime_refit(data = bake_1)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(new_data = bake_1) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=bake_1, by="date") %>%
  select(date, sales)

arima_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = bake_1) %>%
  plot_modeltime_forecast(.interactive=FALSE)


plot1 <- cv_results %>%
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = bake_1) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plot2 <-es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = bake_1) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plot3 <- cv_results %>%
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plot4 <-es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(plot1,plot3,plot2,plot4, nrows = 2)