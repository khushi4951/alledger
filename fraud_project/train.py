import joblib

import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.decomposition
import sklearn.metrics
import sklearn.model_selection

from data_preprocessing import load_and_preprocess_data

def train_models():
    print("Loading data...")
    data = load_and_preprocess_data('creditcard.csv')
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # PCA 2D compression for visualization
    print("Fitting PCA...")
    pca = sklearn.decomposition.PCA(n_components=2, random_state=42)
    X_pca_train = pca.fit_transform(X_train)

    # Initialize all models covered in the course
    models = {
        'Logistic Regression': sklearn.linear_model.LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Decision Tree': sklearn.tree.DecisionTreeClassifier(
            class_weight='balanced', random_state=42, max_depth=10
        ),
        'Random Forest': sklearn.ensemble.RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Naive Bayes': sklearn.naive_bayes.GaussianNB(),
        'KNN': sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        )
    }

    results = {}
    cv_scores = {}
    cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Training models and calculating metrics...")
    for name, model in models.items():
        print(f"-> Fitting {name}")
        model.fit(X_train, y_train)

        # 5-fold cross validation on train set
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        cv_scores[name] = f"{scores.mean().round(4)} +/- {scores.std().round(4)}"

        # Evaluate on test set
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': sklearn.metrics.accuracy_score(y_test, preds),
            'precision': sklearn.metrics.precision_score(y_test, preds),
            'recall': sklearn.metrics.recall_score(y_test, preds),
            'f1': sklearn.metrics.f1_score(y_test, preds),
            'roc_auc': sklearn.metrics.roc_auc_score(y_test, probas),
            'preds': preds,
            'probas': probas
        }

    # Package objects for Streamlit usage
    artifacts = {
        'models': models,
        'results': results,
        'cv_scores': cv_scores,
        'pca': pca,
        'X_pca_train': X_pca_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'amount_scaler': data['amount_scaler'],
        'time_scaler': data['time_scaler']
    }

    joblib.dump(artifacts, 'saved_models.pkl')
    print("\nSaved trained models and evaluation data to saved_models.pkl")

if __name__ == "__main__":
    train_models()