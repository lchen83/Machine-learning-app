library(shiny)
library(shinythemes)
library(DT)
library(tidyverse)  #data manipulation
library(imputeTS) # data imputation
library(caret)
library(dummies)
library(pROC)
library(shinycssloaders)
library(e1071)
library(randomForest)
library(rpart)
library(AppliedPredictiveModeling)

ui <- navbarPage(theme = shinytheme("cerulean"),
                 title = "Machine Learning",
                 tabPanel("Data",
                          tabsetPanel(
                            tabPanel("Data Source",
                                     sidebarLayout(
                                       sidebarPanel(width = 3,
                                                    radioButtons(inputId = "upload", " ", choices = c("Upload Data", "Use sample data"),
                                                                 selected = "Use sample data"),
                                                    conditionalPanel(condition = "input.upload == 'Upload Data'",
                                                                     fileInput(inputId = "file", label = "Choose CSV File",
                                                                               multiple = FALSE,
                                                                               accept = ".csv"),
                                                                     checkboxInput(inputId = "header", label = "Header", TRUE),
                                                                     checkboxInput(inputId = "check", label = "Set the first column as rownames")

                                                    ),
                                                    conditionalPanel(condition = "input.upload == 'Use sample data'",
                                                                     h5("Sample data:"),
                                                                     h5("The Lactose Intolerant data has 970 SNPs data from
                                                                   428 users with two different phenotype: intolerant and tolerant"),
                                                                     downloadButton(outputId = "download", label = "Download sample data")
                                                    )
                                       ),
                                       mainPanel(dataTableOutput(outputId = "df"))
                                     )),
                            tabPanel("Data Preprocess",
                                     sidebarLayout(
                                       sidebarPanel(width = 3,
                                                    h5("Data preprocess"),
                                                    selectInput(inputId = "col_miss", label = "Drop columns with selected percentage of missing values",
                                                                choices = c(100, 90, 80, 70, 60, 50, 40, 30, 20, 10)),
                                                    selectInput(inputId = "row_miss", label = "Drop rows with selected percentage of missing values",
                                                                choices = c(100, 90, 80, 70, 60, 50, 40, 30, 20, 10)),
                                                    # numericInput(inputId = "cor", label = "Correlation cutoff",
                                                    #              min = 0, max = 100, value = 100),
                                                    h5("Split the dataset into training and test"),
                                                    numericInput(inputId = "split", label = "Split ratio:", value = 0.8, width = 100),
                                                    radioButtons(inputId = "impute",
                                                                 label = "Impute missing value with:",
                                                                 choices = c("mean", "median", "mode"),
                                                                 selected = "mode"),
                                                    actionButton(inputId = "act_process", label = "Process"),
                                                    actionButton(inputId = "reset", label = "Reset")),
                                       mainPanel(
                                         dataTableOutput(outputId = "table")
                                       ),
                                     )
                            ))
                 ),
                 tabPanel("SVM",
                          sidebarLayout(
                            sidebarPanel(width = 3,
                                         selectInput(inputId = "svm_target", label = "Choose the Target Variable:",
                                                     choices = ""),
                                         radioButtons(inputId = "svm_predictor", label = "Select predictors",
                                                      choices = c("Use all predictors", "Choose predictors", "PCA"
                                                      ), selected = "PCA"),
                                         conditionalPanel(condition = "input.svm_predictor == 'Choose predictors'",
                                                          selectizeInput(inputId = "svm_choose_predictor",
                                                                         label = "Choose predictors from list",
                                                                         options = list(maxItems = 20),
                                                                         choices = "" )),
                                         conditionalPanel(condition = "input.svm_predictor == 'PCA'",
                                                          numericInput(inputId = "svm_pca_no",
                                                                       label = "Choose the No. of PCs",
                                                                       value = 10, min = 1)),
                                         radioButtons(inputId = "svm_cv", label = "Cross validation",
                                                      choices = c("Yes", "No"), selected = "No"),
                                         conditionalPanel(condition = "input.svm_cv == 'Yes'",
                                                          numericInput(inputId = "svm_cv_fold", label = "No. of fold cross validation:",
                                                                       value = 3, min = 2, max = 10),
                                                          numericInput(inputId = "svm_cv_rep", label = "No. of repeats",
                                                                       value = 3, min = 1, max = 5)
                                         ),
                                         selectInput(inputId = "svm_kernel", label = "Select kernel:",
                                                     choices = c("svmLinear", "svmRadial", "svmPoly"),
                                                     selected = "svmLinear"),
                                         radioButtons(inputId = "svm_tune",
                                                      label = "Do you want to tune model",
                                                      choices = c("Yes", "No"),
                                                      selected = "No"),
                                         conditionalPanel(condition = "input.svm_tune == 'Yes'",
                                                          radioButtons(inputId = "svm_tune_methods",
                                                                       label = "Tune methods",
                                                                       choices = c("Random Search", "Grid Search"),
                                                                       selected = "Random Search")),
                                         conditionalPanel(condition = "input.svm_tune_methods == 'Random Search' &
                                               input.svm_tune == 'Yes'",
                                                          numericInput(inputId = "svm_tune_length",
                                                                       label = "Tune length",
                                                                       value = 5)),
                                         conditionalPanel(condition = "input.svm_tune_methods == 'Grid Search' &
                                                 input.svm_kernel == 'svmLinear' & input.svm_tune == 'Yes'",
                                                          selectizeInput(inputId = "svm_linear_tc", label = "C:",
                                                                         choices = c(1,2,3,4,5),
                                                                         options = list(create = T, maxItems = 5)
                                                          )),
                                         conditionalPanel(condition = "input.svm_tune_methods == 'Grid Search' &
                                                 input.svm_kernel == 'svmRadial' & input.svm_tune == 'Yes'",
                                                          selectizeInput(inputId = "svm_rbf_tc", label = "C:",
                                                                         choices = c(1,2,3,4,5),
                                                                         options = list(create = T, maxItems = 5)),
                                                          selectizeInput(inputId = "svm_rbf_tsigma", label = "sigma:",
                                                                         choices = c(0.11, 0.12, 0.13, 0.14, 0.15),
                                                                         options = list(create = T, maxItems = 5))
                                         ),
                                         conditionalPanel(condition = "input.svm_tune_methods == 'Grid Search' &
                                                 input.svm_kernel == 'svmPoly' & input.svm_tune == 'Yes'",
                                                          selectizeInput(inputId = "svm_poly_tdegree", label = "degree:",
                                                                         choices = c(1, 2, 3),
                                                                         options = list(create = T, maxItems = 5)),
                                                          selectizeInput(inputId = "svm_poly_tscale", label = "scale:",
                                                                         choices = c(0.01, 0.1, 1),
                                                                         options = list(create = T, maxItems = 5)),
                                                          selectizeInput(inputId = "svm_poly_tc", label = "C:",
                                                                         choices = c(1,2,3,4,5),
                                                                         options = list(create = T, maxItems = 5))),
                                         actionButton(inputId = "act_svm", label = "Process")
                            ),
                            mainPanel(
                              tabsetPanel(
                                tabPanel(title = "Confusion matrix",
                                         verbatimTextOutput("svm_cm") %>% withSpinner(type = getOption("spinner.type", default = 7)),
                                         uiOutput("svm_cm_download")
                                ),
                                tabPanel(title = "Model summary",
                                         verbatimTextOutput("svm_fit")),
                                tabPanel(title = "ROC Curve",
                                         plotOutput("svm_roc"))
                              )
                            )
                          ))
)

server <- function(input, output, session){

  #-------------------------------------------data upload-----------------------------------

  data <- reactive({
    if (input$upload == "Use sample data") {
      read.csv("two_label_with_selected_features_rn_v3.csv", row.names = 1)
    } else {
      if (is.null(input$file)) {
        return(NULL)
      } else if (input$check == FALSE) {
        read.csv(input$file$datapath,
                 header = input$header,
                 sep = ",")
      } else if (input$check == TRUE) {
        read.csv(input$file$datapath,
                 header = input$header,
                 sep = ",",
                 row.names = 1)
      }
    }
  })

  output$df <- renderDataTable({
    if (is.null(data())) {
      return()
    } else {
      datatable(data(), options = list(searching = FALSE))
    }
  })

  output$download <- downloadHandler(
    filename = function() {
      paste("Lactose_intolerant_data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(data(), file)
    }
  )

  data_drop_col <- reactive({
    if (length(which(colMeans(is.na(data())) >= as.numeric(input$col_miss) / 100)) == 0) {
      data()
    } else {
      data()[, -which(colMeans(is.na(data())) >= as.numeric(input$col_miss) / 100)]
    }
  })

  clean_data <- reactive({
    if (length(which(rowMeans(is.na(data_drop_col())) >= as.numeric(input$row_miss) / 100)) == 0) {
      data_drop_col()
    } else {
      data_drop_col()[-which(rowMeans(is.na(data_drop_col())) >= as.numeric(input$row_miss) / 100),]
    }
  })

  #------------------------------------------data perprocess-------------------------------------------

  # Defining & initializing the reactiveValues object
  counter <- reactiveValues(countervalue = 0)

  # if the action button is clicked, increment the value by 1 and update it
  observeEvent(input$act_process, {counter$countervalue <- counter$countervalue + 1})

  observeEvent(input$reset, {counter$countervalue <- 0})

  # Remove columns with x% of NAs
  split <- reactive({
    set.seed(100)
    caret::createDataPartition(clean_data()$pheno, p = input$split, list = FALSE)
  })

  train_set <- reactive({
    clean_data()[split(),]
  })

  test_set <- reactive({
    clean_data()[-split(),]
  })

  clean_train <- eventReactive(input$act_process, {
    na_mean(train_set(), option = input$impute)})

  output$table <- renderDataTable({
    datatable(clean_train(), options = list(searching = FALSE), caption = "Training set")
  })

  clean_test <- reactive({na_mean(test_set(), option = input$impute)})

  # standardize data
  train_stand <- reactive({
    standardize_train <- preProcess(clean_train(), method = c("scale", "center"))
    train_scale <- predict(standardize_train, clean_train())
    return(train_scale)
  })

  test_stand <- reactive({
    standardize_test <- preProcess(clean_test(), method = c("scale", "center"))
    train_scale <- predict(standardize_test, clean_test())
    return(train_scale)
  })


  #--------------------------------------------SVM--------------------------------------

  #updata selectizeInput
  observe({
    x <- names(clean_train())

    # Can also set the label and select items
    updateSelectInput(session, "svm_target",
                      choices = x,
                      selected = x[length(x)]
    )
  })

  observe({
    x <- names(clean_train()[, 1:length(clean_train()) - 1])
    updateSelectInput(session, "svm_choose_predictor",
                      choices = x,
                      selected = ""
    )
  })



  # PCA
  pca <- reactive({
    caret::preProcess(train_stand()[, 1:length(train_stand()) - 1],
                      method = 'pca',
                      pcaComp = input$svm_pca_no)
  })

  svm_train_pca <- reactive({predict(pca(), train_stand())})
  svm_test_pca <- reactive({predict(pca(), test_stand())})

  ctrl <- reactive({
    if (input$svm_cv == "Yes" & input$svm_tune == 'Yes') {
      caret::trainControl(method = "repeatedcv",
                          number = input$svm_cv_fold,
                          repeats = input$svm_cv_rep,
                          classProbs = TRUE,
                          search = "random")
    } else if (input$svm_cv == "Yes" & input$svm_tune == 'No') {
      caret::trainControl(method = "repeatedcv",
                          number = input$svm_cv_fold,
                          repeats = input$svm_cv_rep,
                          classProbs = TRUE)
    } else if (input$svm_cv == "No" & input$svm_tune == 'Yes') {
      caret::trainControl(classProbs = TRUE,
                          search = "random")
    } else if (input$svm_cv == "No" & input$svm_tune == 'No') {
      caret::trainControl(classProbs = TRUE)
    }

  })

  grid_search <- reactive({
    if (input$svm_kernel == 'svmLinear') {
      expand.grid(C = as.numeric(input$svm_linear_tc))
    } else if (input$svm_kernel == 'svmRadial') {
      expand.grid(C = as.numeric(input$svm_rbf_tc), sigma = as.numeric(input$svm_rbf_tsigma))
    } else {
      expand.grid(degree = as.numeric(input$svm_poly_tdegree),
                  scale = as.numeric(input$svm_poly_tscale),
                  C = as.numeric(input$svm_poly_tc))
    }
  })

  formula <- reactive({
    if (input$svm_predictor == 'Choose predictors') {
      as.formula(paste(input$svm_target,
                       paste(input$svm_choose_predictor, sep = ",", collapse = " + "),
                       sep = " ~ "))
    } else {
      as.formula(paste(input$svm_target, "~ ."))
    }
  })

  svm_model_data <- reactive({
    if (input$svm_predictor == 'PCA') {
      svm_train_pca()
    } else {
      clean_train()
    }
  })

  svm_test_data <- reactive({
    if (input$svm_predictor == 'PCA') {
      svm_test_pca()
    } else {
      clean_test()
    }
  })

  svm.fit <- eventReactive(input$act_svm, {
    if (input$svm_tune_methods == 'Grid Search' & input$svm_tune == 'Yes') {
      svm <- caret::train(formula(),
                          data = svm_model_data(),
                          method = input$svm_kernel,
                          trControl = ctrl(),
                          tuneGrid = grid_search(),
                          preProcess = c("scale", "center"))
    } else if (input$svm_tune_methods == 'Random Search' & input$svm_tune == 'Yes') {
      svm <- caret::train(formula(),
                          data = svm_model_data(),
                          method = input$svm_kernel,
                          trControl = ctrl(),
                          tuneLength = input$svm_tune_length,
                          preProcess = c("scale", "center"))
    } else if (input$svm_tune == 'No') {
      svm <- caret::train(formula(),
                          data = svm_model_data(),
                          method = input$svm_kernel,
                          trControl = ctrl(),
                          preProcess = c("scale", "center"))
    }
    return(svm)
  })



  svm_con <- eventReactive(input$act_svm, {
    test_pred <- predict(svm.fit(), svm_test_data())
    caret::confusionMatrix(test_pred, svm_test_data()[,input$svm_target], mode =  "everything")
  })

  output$svm_cm <- renderPrint({
    svm_con()
  })

  output$svm_cm_download <- renderUI({
    if (!is.null(svm_con())) {
      downloadButton("svm_cm_outputfile", label = "Download confusion matrix")
    }
  })


  output$svm_cm_outputfile <- shiny::downloadHandler(
    filename = function() {
      paste("svm_confusion_matrix", ".txt")
    },
    content = function(file) {
      sink(file, append = TRUE)
      cat("Confusion Matrix and Statistics", '\n')
      svm_con()[["table"]]
      cat('\n')
      cat("Accuracy : ", svm_con()[["overall"]][["Accuracy"]], '\n')
      cat("95% CI : (", svm_con()[["overall"]][["AccuracyLower"]], ", ", svm_con()[["overall"]][["AccuracyUpper"]], ")", '\n')
      cat("No Information Rate : ", svm_con()[["overall"]][["AccuracyNull"]], '\n')
      cat("P-Value [Acc > NIR] : ", svm_con()[["overall"]][["AccuracyPValue"]], '\n', '\n')
      cat("Kappa : ", svm_con()[["overall"]][["Kappa"]], '\n', '\n')
      cat("Mcnemar's Test P-Value : ", svm_con()[["overall"]][["McnemarPValue"]], '\n', '\n')
      cat("Sensitivity : ", svm_con()[["byClass"]][["Sensitivity"]], '\n')
      cat("Specificity : ", svm_con()[["byClass"]][["Specificity"]], '\n')
      cat("Pos Pred Value : ", svm_con()[["byClass"]][["Pos Pred Value"]], '\n')
      cat("Neg Pred Value : ", svm_con()[["byClass"]][["Neg Pred Value"]], '\n')
      cat("Precision : ", svm_con()[["byClass"]][["Precision"]], '\n')
      cat("Recall : ", svm_con()[["byClass"]][["Recall"]], '\n')
      cat("F1 : ", svm_con()[["byClass"]][["F1"]], '\n')
      cat("Prevalence : ", svm_con()[["byClass"]][["Prevalence"]], '\n')
      cat("Detection Rate : ", svm_con()[["byClass"]][["Detection Rate"]], '\n')
      cat("Detection Prevalence : ", svm_con()[["byClass"]][["Detection Prevalence"]], '\n')
      cat("Balanced Accuracy : ", svm_con()[["byClass"]][["Balanced Accuracy"]], '\n', '\n')
      cat("'Positive' Class : ", svm_con()[["positive"]])
      sink()
    }
  )

  output$svm_fit <- renderPrint({
    svm.fit()
  })

  svm_roc_plot <- eventReactive(input$act_svm, {
    roc_prob <- predict(svm.fit(), svm_test_data(), type = "prob")
    pred <- data.frame(svm_test_data()[, input$svm_target], roc_prob)
    pred <- dummy.data.frame(pred)
    plot.roc(pred[, 1], pred[, 3], print.auc = TRUE)
  })

  output$svm_roc <- renderPlot({
    svm_roc_plot()
  })

}


shinyApp(ui = ui, server = server)
