import pandas as pd
import sklearn
import numpy as np
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def get_labels_from_og(og_labels):
    processed_labels = []
    for og_i, og_label in enumerate(og_labels):
        og_label = og_label[1:-1].replace("'", "")
        if len(og_label) > 0:
            og_label = og_label.split(',')
            processed_labels.append([cc.strip().lower() for cc in og_label])
        else:
            processed_labels.append([])

    return processed_labels


def join_labels(old, df):
    for i, new_label in enumerate(df['added labels']):
        if new_label != '':
            c_new_labels = new_label.split(',')
            c_new_labels = [cc.strip().lower() for cc in c_new_labels]
            old[i] += c_new_labels

    return old


def read_in_csv(fn='./CSSEGISandData _ COVID-19 issues and PR labelling - issues_all.csv'):
    df = pd.read_csv(fn)
    df = df.replace(np.nan, '', regex=True)
    og_labels = df['original labels']
    old_labels = get_labels_from_og(og_labels)
    new_labels = join_labels(old_labels, df)

    return np.asarray(df['title']), new_labels, df


def process_text_to_data_set(titles, labels):
    is_train = [bool(x) for x in labels]
    n_train = sum(is_train)

    # Im already sorting it, so that te separation is easier later
    titles = np.concatenate(
        (titles[is_train], titles[np.logical_not(is_train)]))
    labels = [label for i, label in enumerate(labels) if is_train[i]]

    # TODO check how regex works to get question marks
    vectorizer = feature_extraction.text.TfidfVectorizer(
        min_df=3, max_df=len(titles)//2)
    # If I don't set min_df (minimum document frequency as high,
    # we have more dimensions than samples
    X = vectorizer.fit_transform(titles)

    # I did not find any masking/slice operation for csr matrices, let's see how fast this is ..
    X_train = X[:n_train]
    X_test = X[n_train:]
    unique_labels = np.unique([item for sublist in labels for item in sublist])
    y = np.zeros((n_train, len(unique_labels)))
    for i, label in enumerate(labels):
        y[i] = [def_label in label for def_label in unique_labels]

    return {'labelled': [X_train, y], 'unlabelled': X_test, 'full': X,  'label_names': unique_labels, 'vectorizer': vectorizer}


def train_log_reg_one_vs_all(X, y, label_names):
    classifiers = {}
    for label_i, label in enumerate(label_names):
        clf = LogisticRegression().fit(X, y[:, label_i])
        if sum(y[:, label_i]) > 10:  # If there are more than 10 of this class
            scores = cross_val_score(clf, X, y[:, label_i], cv=5)
            print(
                f"Label {label} -> Accuracy: {scores.mean():0.3f} (+/- {2 * scores.std():0.3f}) [{sum(y[:, label_i])/y.shape[0]:.3f} positive labels]")
        classifiers[label] = clf

    return classifiers


def apply_classifiers_to_data(classifiers, X, title):
    for key in classifiers:
        preds = classifiers[key].predict(X)
        for i, pred in enumerate(preds):
            if pred > .5:
                print(title[i])
                print(f'label "{key}" -> {pred} % ')


if __name__ == "__main__":
    titles, new_labels, df = read_in_csv()
    data_dict = process_text_to_data_set(titles, new_labels)
    class_dict = train_log_reg_one_vs_all(
        *data_dict['labelled'], data_dict['label_names'])
    #apply_classifiers_to_data(class_dict, data_dict['full'], titles)
