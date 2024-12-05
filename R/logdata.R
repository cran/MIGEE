#' @title Longitudinal clinical data for patients
#'
#' @description Longitudinal clinical data including treatment variables and time-to-event outcomes
#' @usage data(logdata)
#' @format A dataframe with multiple rows and 11 variables:
#' \describe{
#' \item{ID}{ID of subjects}
#' \item{Days}{Time in days for each recorded event}
#' \item{Age}{Age of subjects}
#' \item{Gender}{Gender of subjects (Male/Female)}
#' \item{x_val}{Covariate values (numerical)}
#' \item{y_val}{Outcome variable representing time-to-event or measurement (numerical, possibly containing missing data)}
#' \item{trt1}{Treatment group 1 (binary, 0/1)}
#' \item{trt2}{Treatment group 2 (binary, 0/1)}
#' \item{fac1}{Factor 1 (binary, 0/1)}
#' \item{fac2}{Factor 2 (binary, 0/1)}
#' \item{Visit}{Visit number (categorical)}
#' \item{SEX}{Redundant variable for Gender (Male/Female)}
#' }
#' @examples data(logdata)
"logdata"
