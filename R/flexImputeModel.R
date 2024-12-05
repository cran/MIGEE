#' @title Flexible Missing Data Imputation and Statistical Modeling
#'
#' @description
#' This function provides a comprehensive solution for handling missing data, offering flexible imputation methods and
#' advanced modeling options. It allows users to choose how missing values should be imputed, visualize the missingness
#' patterns, and fit both univariate and multivariate models to the data. The function also offers a convenient workflow
#' for splitting datasets and applying user-specified models.
#'
#' @details
#' The function first addresses missing values in specified columns, including `x_val` and `y_val`. Users can select
#' from a range of imputation techniques such as predictive mean matching (`pmm`), k-nearest neighbors (`kNN`),
#' normal linear regression imputation (`norm`), random forest imputation (`rf`), or simple random sampling (`sample`),
#' depending on the nature of their data and the desired analysis.
#'
#' Once missing data is handled, the function splits the dataset into several parts, allowing for more efficient
#' processing or cross-validation. This feature enables users to evaluate imputation and modeling strategies across
#' different portions of the data.
#'
#' The imputation process is highly customizable, letting users specify which variables to impute and which methods
#' to apply. This flexibility ensures that the imputation strategy aligns with the specific requirements of the analysis.
#'
#' After handling missing data, the function transforms the `x_val` variable from a long to a wide format, facilitating
#' modeling of its relationship with `y_val`. A generalized linear model (GLM) is then applied to examine how these
#' variables relate, providing insights into their interaction. Additionally, the function generates heatmaps that
#' offer a visual representation of the missing and non-missing values within the dataset, helping users understand
#' the distribution of their data.
#'
#' For statistical modeling, the function includes options for both univariate and multivariate analysis. It fits
#' linear models (LM) and linear mixed-effects models (LME), allowing users to explore relationships between
#' variables of interest while accounting for random effects if needed. Users can specify which variables to include
#' in the models, making it easy to compare different modeling strategies or adjust for potential confounding variables.
#'
#' @param data A data frame containing the dataset to be used for analysis. It should include columns
#' for the unique ID, time variable, outcome variable, predictor variables, and any other relevant
#' covariates such as age and gender. The data may contain missing values in columns that require imputation.
#' @param id_col A string. The name of the column representing the unique identifier for each subject or observation.
#' @param time_col A string. The name of the column representing time, such as the number of days.
#' @param y_col A string. The name of the outcome or dependent variable column (e.g., "y_val").
#' @param x_col A string. The name of the independent variable column (e.g., "x_val").
#' @param age_col A string. The name of the age column (e.g., "Age").
#' @param gender_col A string. The name of the gender column (e.g., "Gender").
#' @param columns_to_impute A character vector. The names of the columns that have missing values and need imputation (e.g., \code{c("x_val", "y_val")}).
#' If \code{NULL}, all columns with missing data will be imputed.
#' @param methods A character vector. The list of imputation methods to be applied. Defaults to \code{c("pmm", "kNN", "norm", "rf", "norm.nob", "sample")}.
#' @param k An integer. The number of neighbors to use for k-Nearest Neighbors (kNN) imputation. Defaults to 5.
#' @param univariate_vars A character vector. The variables used for univariate analysis. Defaults to \code{c("x_val", "Age")}.
#' @param multivariate_vars A character vector. The variables used for multivariate analysis. Defaults to \code{c("x_val", "Gender")}.
#' @param max_multivariate_vars Maximum number of variables allowed for multivariate analysis is 3.
#' @return A list containing the fitted LM and LME models for both univariate and multivariate analyses, along with generated plots for each method.
#' @import readr mice dplyr reshape2 VIM ggplot2 lme4 ggeffects
#'
#' @references
#' Little, R. J., & Rubin, D. B. (2019). Statistical analysis with missing data (Vol. 793). John Wiley & Sons.
#'
#' @examples
#' \donttest{
#' Results_with_pmm <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("pmm"),
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_pmm$model #summary of Univariate and Multivariate LM and LME model
#' Results_with_kNN <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("kNN"),
#'                           k = 5,
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_kNN$model #summary of Univariate and Multivariate LM and LME model
#' Results_with_norm <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("norm"),
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_norm$model #summary of Univariate and Multivariate LM and LME model
#' Results_with_rf <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("rf"),
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_rf$model #summary of Univariate and Multivariate LM and LME model
#' Results_with_norm.nob <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("norm.nob"),
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_norm.nob$model #summary of Univariate and Multivariate LM and LME model
#' Results_with_sample <- flexImputeModel(data = logdata,
#'                            id_col = "ID",
#'                            time_col = "Days",
#'                            y_col = "y_val",
#'                            x_col = "x_val",
#'                            age_col = "Age",
#'                            gender_col = "Gender",
#'                           columns_to_impute = c("x_val","y_val"),
#'                           methods = c("sample"),
#'                           univariate_vars = c("x_val","Age"),
#'                           multivariate_vars = c("x_val", "Gender", "trt1"),
#'                           max_multivariate_vars = 3)
#' Results_with_sample$model
#' } #summary of Univariate and Multivariate LM and LME model
#' @export
#' @author Atanu Bhattacharjee, Gajendra Kumar Vishwakarma and Neelesh Kumar

#'
flexImputeModel <- function(data, id_col, time_col,
                            y_col, x_col,
                            age_col, gender_col,
                            columns_to_impute = NULL,
                            methods = c('pmm', 'kNN', 'norm', 'rf', 'norm.nob', 'sample'),
                            k = 5, univariate_vars = NULL, multivariate_vars = NULL, max_multivariate_vars = 5) {

  # Ensure that the event column is created based on y_col
  data$event <- ifelse(!is.na(data[[y_col]]), 1, 0)

  # Convert gender and any other character variables in univariate_vars and multivariate_vars to factors
  data[[gender_col]] <- as.factor(data[[gender_col]])

  # Ensure that the `mice` package is loaded
  if (!requireNamespace("mice", quietly = TRUE)) {
    stop("The 'mice' package is required but is not installed. Please install it by running install.packages('mice').")
  }
  if (!requireNamespace("VIM", quietly = TRUE)) {
    stop("The 'VIM' package is required but is not installed. Please install it by running install.packages('VIM').")
  }
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("The 'lme4' package is required but is not installed. Please install it by running install.packages('lme4').")
  }
  if (!requireNamespace("ggeffects", quietly = TRUE)) {
    stop("The 'ggeffects' package is required but is not installed. Please install it by running install.packages('ggeffects').")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required but is not installed. Please install it by running install.packages('ggplot2').")
  }

  # Define the imputation function
  impspt <- function(data, id_col, columns_to_impute = NULL,
                     methods = c('pmm', 'norm', 'rf', 'norm.nob', 'sample'),
                     k = 5) {

    if (is.null(columns_to_impute)) {
      columns_to_impute <- names(data)[colSums(is.na(data)) > 0]
    } else {
      columns_to_impute <- intersect(columns_to_impute, names(data))
    }

    if (length(columns_to_impute) == 0) {
      message("No columns with missing values to impute.")
      return(data)
    }

    data$split_group <- cut(data[[id_col]], breaks = 10, labels = FALSE,
                            include.lowest = TRUE)
    data_splits <- split(data, data$split_group)

    updated_data_list <- list()

    for (method in methods) {
      results <- list()

      for (i in 1:10) {
        data_split <- data_splits[[i]]

        imputed_data <- suppressWarnings({
          if (method == 'kNN') {
            VIM::kNN(data_split, k = k)
          } else {
            mydata_imp <- mice::mice(data_split, method = method, m = 1)
            mice::complete(mydata_imp, action = 1)
          }
        })

        results[[i]] <- imputed_data
      }

      combined_results <- do.call(rbind, results)  # Bind the results without dplyr

      updated_data <- data
      for (i in 1:nrow(updated_data)) {
        for (j in seq_along(columns_to_impute)) {
          col_name <- columns_to_impute[j]
          if (is.na(updated_data[i, col_name])) {
            updated_data[i, col_name] <- combined_results[i, col_name]
          }
        }
      }

      updated_data_list[[method]] <- updated_data
    }

    return(updated_data_list)
  }

  # Impute the data using specified methods and columns
  updated_data <- impspt(data, id_col = id_col,
                         columns_to_impute = columns_to_impute,
                         methods = methods, k = k)

  # Function to fit GLM and plot
  fit_and_plot_glm <- function(data, title) {
    glm_model <- function(data) {
      clean_data <- data[complete.cases(data[, c("event", grep("^x", names(data), value = TRUE))]), ]
      suppressWarnings({
        glm(event ~ ., data = clean_data, family = binomial)
      })
    }

    glm_fit <- glm_model(data)
    summary(glm_fit)

    plot_glm_model <- function(model, data, title) {
      clean_data <- data[complete.cases(data[, c("event", grep("^x", names(data), value = TRUE))]), ]

      preds <- suppressWarnings({
        predict(model, newdata = clean_data, type = "response")
      })

      plot_data <- data.frame(Predicted = preds, Event = clean_data$event)

      p <- plot(plot_data$Predicted, plot_data$Event, main = title, xlab = "Predicted Probability", ylab = "Event")
      lines(lowess(plot_data$Predicted, plot_data$Event), col = "blue")

      print(p)  # Ensure the plot is displayed
    }

    plot_glm_model(glm_fit, data, title)
    return(glm_fit)
  }

  # Fit and plot GLM for each imputed dataset
  names(updated_data) <- methods
  glm_fits <- list()

  for (method in methods) {
    glm_fit <- fit_and_plot_glm(updated_data[[method]], paste("GLM with", method, "Imputation"))
    glm_fits[[method]] <- glm_fit
  }

  # Creating the heatmap visualization for missingness
  data_for_heatmap <- data[, grep("^x", names(data))]

  if (any(colSums(is.na(data_for_heatmap)) > 0)) {
    suppressWarnings({
      VIM::aggr(data_for_heatmap, col=c('navyblue','red'),
                numbers=TRUE, sortVars=TRUE,
                labels=names(data_for_heatmap),
                cex.axis=.7, gap=3,
                ylab=c("Missing data","Pattern"))
    })
  } else {
    message("No missing values in the selected columns.")
  }

  # Function to fit lm and lme models
  fit_models <- function(data, univariate_vars, multivariate_vars, max_multivariate_vars) {
    lm_models <- list()
    lme_models <- list()
    lm_summaries <- list()
    lme_summaries <- list()

    # Univariate models
    if (!is.null(univariate_vars)) {
      for (var in univariate_vars) {
        lm_model <- suppressWarnings({
          lm(as.formula(paste(y_col, "~", var)), data = data)
        })
        lm_summaries[[var]] <- summary(lm_model)

        lme_model <- suppressWarnings({
          lme4::lmer(as.formula(paste(y_col, "~", var, "+ (1 |", id_col, ")")), data = data)
        })
        lme_summaries[[var]] <- summary(lme_model)

        lm_models[[var]] <- lm_model
        lme_models[[var]] <- lme_model
      }
    }

    # Multivariate models - allowing the user to choose between 1 to max_multivariate_vars variables
    if (!is.null(multivariate_vars)) {
      selected_vars <- multivariate_vars[1:min(length(multivariate_vars), max_multivariate_vars)]

      lm_model <- suppressWarnings({
        lm(as.formula(paste(y_col, "~", paste(selected_vars, collapse = " + "))), data = data)
      })
      lm_summaries[["multivariate"]] <- summary(lm_model)

      lme_model <- suppressWarnings({
        lme4::lmer(as.formula(paste(y_col, "~", paste(selected_vars, collapse = " + "), "+ (1 |", id_col, ")")), data = data)
      })
      lme_summaries[["multivariate"]] <- summary(lme_model)

      lm_models[["multivariate"]] <- lm_model
      lme_models[["multivariate"]] <- lme_model
    }

    list(lm_models = lm_models, lme_models = lme_models, lm_summaries = lm_summaries, lme_summaries = lme_summaries)
  }

  # Fit the models to the imputed dataset using the first imputation method as an example
  models <- fit_models(updated_data[[methods[1]]], univariate_vars, multivariate_vars, max_multivariate_vars)

  # Create ggpredict plots for the lm and lme models
  lm_plots <- list()
  lme_plots <- list()

  for (var in names(models$lm_models)) {
    if (var == "multivariate") {
      lm_plot <- suppressWarnings({
        ggeffects::ggpredict(models$lm_models[[var]], terms = multivariate_vars) |>
          plot() + ggplot2::ggtitle("Multivariate LM: Predicted values based on selected variables")
      })

      lme_plot <- suppressWarnings({
        ggeffects::ggpredict(models$lme_models[[var]], terms = multivariate_vars) |>
          plot() + ggplot2::ggtitle("Multivariate LME: Predicted values based on selected variables")
      })
    } else {
      lm_plot <- suppressWarnings({
        ggeffects::ggpredict(models$lm_models[[var]], terms = var) |>
          plot() + ggplot2::ggtitle(paste("Univariate LM: Predicted", y_col, "based on", var))
      })

      lme_plot <- suppressWarnings({
        ggeffects::ggpredict(models$lme_models[[var]], terms = var) |>
          plot() + ggplot2::ggtitle(paste("Univariate LME: Predicted", y_col, "based on", var))
      })
    }

    lm_plots[[var]] <- lm_plot
    lme_plots[[var]] <- lme_plot
  }

  # Print the plots
  for (var in names(lm_plots)) {
    print(lm_plots[[var]])
    print(lme_plots[[var]])
  }

  return(list(models = models, plots = list(lm_plots = lm_plots, lme_plots = lme_plots), glm_fits = glm_fits))
}


utils::globalVariables(c('complete.cases','lowess',"kNN","mice","complete","par","density","lines","bind_rows","%>%","step","select","event","starts_with","na.omit","glm","binommial","%>%","select","event","start_with","na.omit","predict","ggplot","aes","predicted","Event","geom_point","geom_smooth","binomial","labs","theme_minimal","%>%","select","starts_with","aggr","lm","as.formula","lmer","ggpredict","ggtitle","read.csv","Predicted"))
