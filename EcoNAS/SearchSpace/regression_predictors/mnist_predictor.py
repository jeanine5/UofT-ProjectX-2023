"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class MNISTBenchmark:
    def __init__(self, csv_filename='../SearchSpace/precomputed_datasets/trained_mnist_datasetcsv'):
        self.df = pd.read_csv(csv_filename)
        self.X = self.df[['hidden_layers', 'hidden_sizes_mean', 'max_hidden_size', 'min_hidden_size',
                          'variance_hidden_sizes']]

        self.y_accuracy = self.df['accuracy']
        self.y_interpretability = self.df['introspectability']
        self.y_flops = self.df['flops']

        self.X_train, self.X_test, self.y_acc_train, self.y_acc_test, self.y_int_train, self.y_int_test, self.y_flops_train, self.y_flops_test = train_test_split(
            self.X, self.y_accuracy, self.y_interpretability, self.y_flops, test_size=0.2, random_state=42)

        self.regression_model = DecisionTreeRegressor()

        self.model_accuracy = DecisionTreeRegressor()
        self.model_interpretability = DecisionTreeRegressor()
        self.model_flops = DecisionTreeRegressor()

    def train_models(self):
        """
        Train the linear regression model after preprocessing the data
        """
        self.model_accuracy.fit(self.X_train, self.y_acc_train)
        self.model_interpretability.fit(self.X_train, self.y_int_train)
        self.model_flops.fit(self.X_train, self.y_flops_train)

    def evaluate_models(self):
        """
        Evaluate the performance of the regression model on the test set
        """
        y_acc_pred = self.model_accuracy.predict(self.X_test)
        y_int_pred = self.model_interpretability.predict(self.X_test)
        y_flops_pred = self.model_flops.predict(self.X_test)

        mean_acc_pred = y_acc_pred.mean()
        mean_int_pred = y_int_pred.mean()
        mean_flops_pred = y_flops_pred.mean()

        return mean_acc_pred, mean_int_pred, mean_flops_pred

    def predict_performance(self, new_architecture):
        """
        Make predictions for the performance of a new architecture
        """
        hidden_sizes = new_architecture.hidden_sizes
        num_hidden_layers = len(hidden_sizes)
        hidden_sizes_mean = np.mean(hidden_sizes)

        # Prepare input data for prediction
        new_data = {'hidden_layers': [num_hidden_layers], 'hidden_sizes_mean': [hidden_sizes_mean],
                    'max_hidden_size': max(new_architecture.hidden_sizes),
                    'min_hidden_size': min(new_architecture.hidden_sizes),
                    'variance_hidden_sizes': np.var(new_architecture.hidden_sizes)}

        # Use trained regression models to predict performance metrics
        acc_pred = self.model_accuracy.predict(pd.DataFrame(new_data, columns=self.X.columns))
        int_pred = self.model_interpretability.predict(pd.DataFrame(new_data, columns=self.X.columns))
        flops_pred = self.model_flops.predict(pd.DataFrame(new_data, columns=self.X.columns))

        return acc_pred.mean(), int_pred.mean(), flops_pred.mean()

