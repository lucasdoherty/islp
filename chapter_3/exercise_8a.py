"""Chapter 3 | Linear Regression pplied exercises."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import summarize


def main():
    """Run Python scipt."""
    # (a) Use the sm.OLS() function to perform a simple linear regression with
    # mpg as the response and horsepower as the predictor. Use the summarize()
    # function to print the results. Comment on the output. For example:
    #
    # i.   Is there a relationship between the predictor and the response?
    # ii.  How strong is the relationship between the predictor and the
    #      response?
    # iii. Is the relationship between the predictor and the response positive
    #      or negative?
    # iv.  What is the predicted mpg associated with a horsepower of 98? What
    #      are the associated 95% confidence and prediction intervals?

    auto_data = load_data("Auto")
    X = pd.DataFrame(
        {
            "intercept": np.ones(auto_data.shape[0]),
            "horsepower": auto_data["horsepower"],
        }
    )
    y = auto_data["mpg"]
    model = sm.OLS(y, X)
    results = model.fit()
    print(summarize(results))

    # i. There appears to be a statistically significant relationship between
    # horsepower and mpg due to a p-value of 0 for the horsepower coefficient.

    # ii. The large negative coefficient and high t-value for the predictor
    # suggests a strong negative relationship between horsepower and mpg.

    # iii. The relationship is negative as can be seen by the negative sign
    # in the coefficient for the predictor.

    # iv. mpg = 39.9359 − 0.1578 × 98 = 39.9359 − 15.4644 = 24.4715


if __name__ == "__main__":
    main()
