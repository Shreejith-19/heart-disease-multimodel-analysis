
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
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
    def test_model(self,trained_model, x_train, y_train, x_test, y_test):
        """
        """
        print("Training Results: ")
        print(classification_report(y_train, trained_model.predict(x_train)))
        print("Test Results :")
        print(classification_report(y_test, trained_model.predict(x_test)))
        print("test_accuracy = ",accuracy_score(trained_model.predict(x_test), y_test))

