from sklearn import preprocessing

__all__ = [
    'encodeLabels'
]

def encodeLabels(labels):
    le = preprocessing.LabelEncoder()
    set_labels = list(set(labels))
    le.fit(labels)
    return dict(zip(set_labels, le.transform(set_labels)))
