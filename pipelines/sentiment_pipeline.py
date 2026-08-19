from zenml import pipeline

from steps.data_ingestion import ingest_data
from steps.preprocess_data import preprocess_data
from steps.feature_engineering import create_tfidf_features

from steps.train_random_forest import train_random_forest
from steps.train_svm import train_svm
from steps.train_naive_bayes import train_naive_bayes

from steps.save_best_model import save_best_model


@pipeline
def sentiment_pipeline():

    # ============================================================
    # 1. DATA INGESTION
    # ============================================================

    train_data, test_data = ingest_data()

    # ============================================================
    # 2. DATA PREPROCESSING
    # ============================================================

    X_train, X_test, y_train, y_test = preprocess_data(
        train_df=train_data,
        test_df=test_data,
    )

    # ============================================================
    # 3. TF-IDF FEATURE ENGINEERING
    # ============================================================

    (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer,
    ) = create_tfidf_features(
        X_train=X_train,
        X_test=X_test,
    )

    # ============================================================
    # 4. RANDOM FOREST
    # ============================================================

    (
        random_forest_model,
        random_forest_f1,
    ) = train_random_forest(
        X_train=X_train_tfidf,
        X_test=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
    )

    # ============================================================
    # 5. SVM
    # ============================================================

    (
        svm_model,
        svm_f1,
    ) = train_svm(
        X_train=X_train_tfidf,
        X_test=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
    )

    # ============================================================
    # 6. NAIVE BAYES
    # ============================================================

    (
        naive_bayes_model,
        naive_bayes_f1,
    ) = train_naive_bayes(
        X_train=X_train_tfidf,
        X_test=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
    )

    # ============================================================
    # 7. SELECT AND SAVE BEST MODEL
    # ============================================================

    best_model_path = save_best_model(
        random_forest_model=random_forest_model,
        random_forest_f1=random_forest_f1,

        svm_model=svm_model,
        svm_f1=svm_f1,

        naive_bayes_model=naive_bayes_model,
        naive_bayes_f1=naive_bayes_f1,

	vectorizer=vectorizer,

    )

    return best_model_path


# ================================================================
# RUN PIPELINE
# ================================================================

if __name__ == "__main__":
    sentiment_pipeline()



