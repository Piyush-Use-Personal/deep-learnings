# main.py
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model_training import train_models
from evaluation import evaluate_models
from visualization import plot_roc_curve, plot_confusion_matrix

def main():
    # Load and preprocess data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    results = evaluate_models(trained_models, X_test, y_test)
    print("Model Evaluation Results:")
    print(pd.DataFrame(results).T)

    # Visualize results
    plot_roc_curve(trained_models, X_test, y_test)
    plot_confusion_matrix(trained_models, X_test, y_test)

if __name__ == "__main__":
    main()