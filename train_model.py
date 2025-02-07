import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a posture detection model using a synthetic dataset."
    )
    parser.add_argument(
        "--exercise",
        type=str,
        required=True,
        help=("Exercise name (choose one of: shoulder_elbow_flexion, pendulum, crossover_arm_stretch). "
              "The script will look for a CSV file named <exercise>_dataset.csv unless --dataset is provided.")
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the CSV dataset file. If not provided, uses <exercise>_dataset.csv in the current directory."
    )
    args = parser.parse_args()

    # Determine CSV filename
    csv_file = args.dataset if args.dataset else f"{args.exercise}_dataset.csv"
    print(f"Loading dataset from: {csv_file}")

    # Load dataset using pandas
    data = pd.read_csv(csv_file)
    print("Dataset preview:")
    print(data.head())

    # Define the feature columns and the label column
    features = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "left_shoulder_angle",
        "right_shoulder_angle"
    ]
    label = "correct_posture"

    # Extract features (X) and label (y)
    X = data[features]
    y = data[label]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the Random Forest classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    model_filename = f"{args.exercise}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved to: {model_filename}")

if __name__ == "__main__":
    main()
