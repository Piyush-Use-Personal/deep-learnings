import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

class CEDL_Structured:
    def __init__(self, dataset_path, target_column, threshold=None):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.threshold = threshold
        self.models = []
        self.optimizers = {
            'SGD': SGD(),
            'RMSprop': RMSprop(),
            'Adagrad': Adagrad(),
            'Adadelta': Adadelta(),
            'Adam': Adam(),
            'Adamax': Adamax(),
            'Nadam': Nadam()
        }
        self.weights = {}
        self.accuracies = {}
        
    def load_data(self):
        """Load and preprocess the dataset"""
        data = pd.read_csv(self.dataset_path)
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self, optimizer, input_shape):
        """Create a simple deep learning model"""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models with different optimizers"""
        for opt_name, optimizer in self.optimizers.items():
            print(f"\nTraining model with {opt_name} optimizer...")
            model = self.create_model(optimizer, X_train.shape[1])
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                               validation_data=(X_test, y_test), verbose=0)
            
            # Evaluate the model
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.accuracies[opt_name] = accuracy
            self.models.append((opt_name, model))
            
            print(f"{opt_name} Accuracy: {accuracy:.4f}")
    
    def calculate_weights(self):
        """Calculate weights for each model based on accuracy"""
        total_weight = sum(self.accuracies.values())
        for opt_name, accuracy in self.accuracies.items():
            self.weights[opt_name] = accuracy / total_weight
        
        # If threshold is not provided, calculate as average weight
        if self.threshold is None:
            self.threshold = sum(self.weights.values()) / len(self.weights)
        
        print("\nModel Weights:")
        for opt_name, weight in self.weights.items():
            print(f"{opt_name}: {weight:.4f}")
        print(f"\nThreshold: {self.threshold:.4f}")
    
    def ensemble_prediction(self, X):
        """Make predictions using the ensemble of selected models"""
        predictions = []
        selected_models = []
        
        # Select models with weight >= threshold
        for opt_name, model in self.models:
            if self.weights[opt_name] >= self.threshold:
                selected_models.append((opt_name, model))
                pred = model.predict(X).flatten()
                weighted_pred = pred * self.weights[opt_name]
                predictions.append(weighted_pred)
        
        print("\nSelected Models for Ensemble:")
        for opt_name, _ in selected_models:
            print(f"{opt_name} (Weight: {self.weights[opt_name]:.4f})")
        
        # Average the predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return np.round(ensemble_pred)
    
    def run(self):
        """Run the complete CEDL pipeline"""
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_models(X_train, y_train, X_test, y_test)
        self.calculate_weights()
        
        # Make ensemble predictions
        y_pred = self.ensemble_prediction(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")
        
        return {
            'individual_accuracies': self.accuracies,
            'weights': self.weights,
            'threshold': self.threshold,
            'ensemble_accuracy': ensemble_accuracy
        }

# =======================
# Example usage
# =======================
if __name__ == "__main__":
    # Ensure 'heart_disease.csv' is in your current working directory
    cedl = CEDL_Structured(dataset_path='heart_disease.csv', 
                           target_column='target',
                           threshold=0.1)  # or None to use average weight
    
    results = cedl.run()
