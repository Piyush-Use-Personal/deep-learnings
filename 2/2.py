import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

class CEDL_Unstructured:
    def __init__(self, dataset_path, text_column, label_column, max_words=10000, max_len=100):
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.label_column = label_column
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.models = []
        self.accuracies = {}
        
    def load_data(self):
        """Load and preprocess the dataset"""
        data = pd.read_csv(self.dataset_path)
        texts = data[self.text_column].values
        labels = data[self.label_column].values
        
        # Tokenize text
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def create_cnn_model(self):
        """Create CNN model for text classification"""
        model = Sequential()
        model.add(Embedding(self.max_words, 128, input_length=self.max_len))
        model.add(Dropout(0.2))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def create_gru_model(self):
        """Create GRU model for text classification"""
        model = Sequential()
        model.add(Embedding(self.max_words, 128, input_length=self.max_len))
        model.add(Dropout(0.2))
        model.add(GRU(128))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def create_lstm_model(self):
        """Create LSTM model for text classification"""
        model = Sequential()
        model.add(Embedding(self.max_words, 128, input_length=self.max_len))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all three competing models"""
        models = {
            'CNN': self.create_cnn_model(),
            'GRU': self.create_gru_model(),
            'LSTM': self.create_lstm_model()
        }
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            history = model.fit(X_train, y_train, epochs=5, batch_size=64, 
                               validation_data=(X_test, y_test), verbose=1)
            
            # Evaluate the model
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.accuracies[name] = accuracy
            self.models.append((name, model))
            
            print(f"{name} Accuracy: {accuracy:.4f}")
    
    def select_best_model(self):
        """Select the model with highest accuracy"""
        best_model_name = max(self.accuracies, key=self.accuracies.get)
        best_model = next(model for name, model in self.models if name == best_model_name)
        best_accuracy = self.accuracies[best_model_name]
        
        print(f"\nSelected best model: {best_model_name} with accuracy {best_accuracy:.4f}")
        return best_model
    
    def run(self):
        """Run the complete CEDL pipeline for unstructured data"""
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_models(X_train, y_train, X_test, y_test)
        best_model = self.select_best_model()
        
        # Evaluate best model
        y_pred = (best_model.predict(X_test) > 0.5).astype(int)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")
        
        return {
            'model_accuracies': self.accuracies,
            'best_model': best_model,
            'final_accuracy': final_accuracy
        }

# Example usage
if __name__ == "__main__":
    # Example with a sentiment analysis dataset (replace with your actual dataset)
    cedl_text = CEDL_Unstructured(dataset_path='twitter_sentiment.csv',
                                 text_column='tweet',
                                 label_column='sentiment')
    
    results = cedl_text.run()