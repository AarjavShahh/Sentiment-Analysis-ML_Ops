from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Annotated
from zenml import step


@step
def create_tfidf_features(
    X_train,
    X_test,
) -> tuple[
    Annotated[object, "X_train_tfidf"],
    Annotated[object, "X_test_tfidf"],
    Annotated[TfidfVectorizer, "vectorizer"],
]:

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train TF-IDF:", X_train_tfidf.shape)
    print("Test TF-IDF:", X_test_tfidf.shape)

    return X_train_tfidf, X_test_tfidf, vectorizer


