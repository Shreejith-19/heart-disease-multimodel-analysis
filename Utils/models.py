
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
class Models:
    def train_random_forest_model(self,x_train, y_train, rdf_model, params_grid):
        """
        """
        grid_search = GridSearchCV(rdf_model , params_grid, scoring = "f1_macro", cv = 5, n_jobs = 5)
        grid_search.fit(x_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_SVM_model(self, model, x_train, y_train):
        """
        """
        model.fit(x_train, y_train)
    def train_logistic_model(self):
        """
        """
    def train_Neural_Network_model(self):
        """
        """
    def test_model(self,trained_model, x_train, y_train, x_test, y_test, type):
        """
        """
        def eval(y_pred, test_type):
            print(f"{test_type} Results:")
            print(f"accuracy score = {accuracy_score(y_pred, y_train)}")
            print(f"Macro-F1 = {f1_score(y_pred, y_train, average = "macro")}")
            print(f"Weighted-F1 = {f1_score(y_pred, y_train, average = "weighted")}")

        if type == "ANN":
            y_pred_class_train = np.argmax(trained_model.predict(x_train), axis = 1)
            y_pred_class_test = np.argmax(trained_model.predict(x_test), axis = 1)
            eval(y_pred_class_train, "Training")
            print("\n")
            eval(y_pred_class_test, "Testing")
        else:
            y_pred_train = trained_model.predict(x_train)
            y_pred_test = trained_model.predict(x_test)
            eval(y_pred_train, "Training")
            print("\n")
            eval(y_pred_test, "Testing")
