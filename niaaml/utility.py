from sklearn import preprocessing

__all__ = [
    'get_label_encoder'
]

def get_label_encoder(labels):
    r"""Get the label encoder instance.

    Arguments:
        labels (Iterable[string]): Array of labels.

    Returns:
        sklearn.preprocessing.LabelEncoder: Instance of a LabelEncoder fit to the given labels.
    """

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le
