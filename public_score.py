from sklearn.metrics import confusion_matrix

def cohen_kappa_score(y1, y2, labels=None, weights=None):
    confusion = confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1).astype(np.double) / np.sum(sum0) # type changed to double for consistency 
    
    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    
    return 1 - k

def cohen_kappa_score_function(y_true, y_pred):
    return cohen_kappa_score(y_pred, y_true)

