library(tseries)
library(forecast)
library(rugarch)
library(zoo)

start_date <- as.Date("2010-01-01")
end_date   <- as.Date("2025-12-31")


prices.dat <- get.hist.quote(instrument = "^FTMC",
                             start = start_date,
                             end = end_date,
                             quote = "AdjClose",
                             compression = "d")

prices.dat <- na.omit(prices.dat)

log.returns <- diff(log(prices.dat))

log.numeric <- as.numeric(log.returns)

plot(prices.dat,
     ylab = "Raw Price",
     xlab = "Date",
     main = "FTSE 250 Daily Prices")

plot(log.returns,
     ylab = "Log Returns",
     xlab = "Date",
     main = "FTSE 250 Daily Log Returns")


adf.test(as.numeric(prices.dat))
adf.test(log.numeric)

acf(log.numeric,
    main = "ACF: FTSE 250 Log Returns")

acf(log.numeric^2,
    main = "ACF: Squared FTSE 250 Log Returns")

Box.test(log.numeric,
         lag = 10,
         type = "Ljung-Box")

Box.test(log.numeric^2,
         lag = 10,
         type = "Ljung-Box")

arma.fit <- auto.arima(log.numeric,
                       max.p = 9,
                       max.q = 9,
                       ic = "bic",
                       approximation = FALSE)

arma.fit


arma.resid <- as.numeric(residuals(arma.fit))

acf(arma.resid,
    main = "ACF: AR(1) Residuals")

acf(arma.resid^2,
    main = "ACF: Squared AR(1) Residuals")

Box.test(arma.resid,
         lag = 10,
         type = "Ljung-Box",
         fitdf = 1)

Box.test(arma.resid^2,
         lag = 10,
         type = "Ljung-Box")

spec.norm <- ugarchspec(
  variance.model = list(model = "sGARCH",
                        garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0)),
  distribution.model = "norm"
)

fit.norm <- ugarchfit(spec = spec.norm,
                      data = log.numeric)

fit.norm

resid.garch.norm <- as.numeric(residuals(fit.norm,
                                         standardize = TRUE))

acf(resid.garch.norm,
    main = "ACF: Standardised Residuals, AR(1)-GARCH(1,1)")

acf(resid.garch.norm^2,
    main = "ACF: Squared Standardised Residuals, AR(1)-GARCH(1,1)")

Box.test(resid.garch.norm,
         lag = 10,
         type = "Ljung-Box",
         fitdf = 1)

Box.test(resid.garch.norm^2,
         lag = 10,
         type = "Ljung-Box")

qqnorm(resid.garch.norm,
       main = "Normal Q-Q Plot")

qqline(resid.garch.norm,
       col = "blue",
       lwd = 2)

spec.t <- ugarchspec(
  variance.model = list(model = "sGARCH",
                        garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0)),
  distribution.model = "std"
)

fit.t <- ugarchfit(spec = spec.t,
                   data = log.numeric)

fit.t

resid.t <- as.numeric(residuals(fit.t,
                                standardize = TRUE))

theory.t <- qdist("std",
                  p = ppoints(length(resid.t)),
                  shape = coef(fit.t)["shape"])

qqplot(theory.t,
       resid.t,
       main = "Student-t Q-Q Plot",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles")

abline(0, 1,
       col = "blue",
       lwd = 2)

spec.sstd <- ugarchspec(
  variance.model = list(model = "sGARCH",
                        garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0)),
  distribution.model = "sstd"
)

fit.sstd <- ugarchfit(spec = spec.sstd,
                      data = log.numeric)

fit.sstd

resid.sstd <- as.numeric(residuals(fit.sstd,
                                   standardize = TRUE))

theory.sstd <- qdist("sstd",
                     p = ppoints(length(resid.sstd)),
                     skew = coef(fit.sstd)["skew"],
                     shape = coef(fit.sstd)["shape"])

qqplot(theory.sstd,
       resid.sstd,
       main = "Skewed Student-t Q-Q Plot",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles")

abline(0, 1,
       col = "blue",
       lwd = 2)


acf(resid.sstd,
    main = "ACF: Skewed-t AR(1)-GARCH(1,1)")

acf(resid.sstd^2,
    main = "ACF: Squared Skewed-t AR(1)-GARCH(1,1)")

Box.test(resid.sstd,
         lag = 10,
         type = "Ljung-Box",
         fitdf = 1)

Box.test(resid.sstd^2,
         lag = 10,
         type = "Ljung-Box")

Rt_squared <- log.numeric[-1]^2
Rt_lagged <- log.numeric[-length(log.numeric)]

leverage.cor <- cor(Rt_squared, Rt_lagged)
leverage.cor

abs_after_pos <- abs(log.numeric[-1][Rt_lagged > 0])
abs_after_neg <- abs(log.numeric[-1][Rt_lagged < 0])

qqplot(abs_after_pos,
       abs_after_neg,
       main = "Leverage Effect",
       xlab = "Quantiles of |Rt| after Positive Returns",
       ylab = "Quantiles of |Rt| after Negative Returns")

abline(0, 1,
       col = "blue",
       lwd = 2)

spec.egarch <- ugarchspec(
  variance.model = list(model = "eGARCH",
                        garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0)),
  distribution.model = "sstd"
)

fit.egarch <- ugarchfit(spec = spec.egarch,
                        data = log.numeric)

fit.egarch

resid.egarch <- as.numeric(residuals(fit.egarch,
                                     standardize = TRUE))

Box.test(resid.egarch,
         lag = 10,
         type = "Ljung-Box",
         fitdf = 1)

Box.test(resid.egarch^2,
         lag = 10,
         type = "Ljung-Box")

spec.aparch <- ugarchspec(
  variance.model = list(model = "apARCH",
                        garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 0)),
  distribution.model = "sstd"
)

fit.aparch <- ugarchfit(spec = spec.aparch,
                        data = log.numeric)

fit.aparch

resid.aparch <- as.numeric(residuals(fit.aparch,
                                     standardize = TRUE))

Box.test(resid.aparch,
         lag = 10,
         type = "Ljung-Box",
         fitdf = 1)

Box.test(resid.aparch^2,
         lag = 10,
         type = "Ljung-Box")

#Lets say we are investing 10000

P <- 10000

forecast.egarch <- ugarchforecast(fit.egarch,
                                  n.ahead = 300)

sigma.forecast <- as.numeric(sigma(forecast.egarch))
mu.forecast <- as.numeric(fitted(forecast.egarch))

q.sstd <- qdist("sstd",
                p = 0.01,
                skew = coef(fit.egarch)["skew"],
                shape = coef(fit.egarch)["shape"])

VaR.forecast <- -P * (mu.forecast +
                        q.sstd * sigma.forecast)

VaR.forecast

max(VaR.forecast)
which.max(VaR.forecast)

sigma.forecast[300]

plot(1:300,
     sigma.forecast,
     type = "l",
     lwd  = 2,
     xlab = "Days Ahead",
     ylab = "Conditional Volatility",
     main = "300-Day Forecast of Conditional Volatility")
