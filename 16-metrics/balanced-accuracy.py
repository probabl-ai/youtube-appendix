import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    import numpy as np 
    import marimo as mo
    import altair as alt
    from wigglystuff import Matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    return (
        LogisticRegression,
        Matrix,
        SVC,
        accuracy_score,
        alt,
        f1_score,
        make_classification,
        mo,
        np,
        pl,
        precision_score,
        recall_score,
        train_test_split,
    )


@app.cell
def __(LogisticRegression, make_classification, train_test_split):
    X, y = make_classification(5000, n_informative=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    mod = LogisticRegression().fit(X_train, y_train)
    return X, X_test, X_train, mod, y, y_test, y_train


@app.cell
def __(X_test, accuracy_score, mod, np, y_test):
    preds = mod.predict(X_test)
    np.mean(preds == y_test), accuracy_score(y_test, preds)
    return (preds,)


@app.cell
def __(np, precision_score, preds, y_test):
    when_precision = (preds == 1)
    np.mean(preds[when_precision] == y_test[when_precision]), precision_score(y_test, preds)
    return (when_precision,)


@app.cell
def __(np, preds, recall_score, y_test):
    when_recall = (y_test == 1)
    np.mean(preds[when_recall] == y_test[when_recall]), recall_score(y_test, preds)
    return (when_recall,)


@app.cell
def __(f1_score, precision_score, preds, recall_score, y_test):
    pre = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    2 * (pre * rec)/(pre + rec), f1_score(y_test, preds)
    return pre, rec


@app.cell
def __(
    X_test,
    accuracy_score,
    alt,
    f1_score,
    mod,
    np,
    pl,
    precision_score,
    recall_score,
    y_test,
):
    data = []
    for threshold in np.linspace(0.01, 0.99, 200):
        t_preds = mod.predict_proba(X_test)[:, 1] > threshold
        data.append({
            "threshold": threshold, 
            "accuracy": accuracy_score(y_test, t_preds), 
            "precision": precision_score(y_test, t_preds),
            "recall": recall_score(y_test, t_preds),
            "f1-score": f1_score(y_test, t_preds),
        })

    pltr = pl.DataFrame(data).unpivot(index="threshold")

    alt.Chart(pltr).mark_line().encode(x="threshold", y="value", color="variable").properties(width=800).interactive()
    return data, pltr, t_preds, threshold


@app.cell
def __():
    from sympy import symbols, solve, expand, simplify

    # Define variables for confusion matrix
    tp, tn, fp, fn = symbols('tp tn fp fn')

    # Basic metrics 
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Try to solve system of equations
    solution = solve([
        fp - fn
    ])


    print("Solution space:")
    print(solution)

    print("\nMetrics in terms of remaining variables:")
    print(f"Precision = {simplify(precision.subs(solution))}")
    print(f"Recall = {simplify(recall.subs(solution))}")
    print(f"F1 Score = {simplify(f1.subs(solution))}")
    return (
        accuracy,
        expand,
        f1,
        fn,
        fp,
        precision,
        recall,
        simplify,
        solution,
        solve,
        symbols,
        tn,
        tp,
    )


@app.cell
def __():
    # data_atoms = []
    # for t in np.linspace(0.01, 0.99, 200):
    #     a_preds = mod.predict_proba(X_test)[:, 1] > t
    #     tn, fp, fn, tp = confusion_matrix(y_test, a_preds).ravel()
    #     data_atoms.append({
    #         "threshold": t, 
    #         "tn": tn,
    #         "fp": fp,
    #         "fn": fn,
    #         "tp": tp,
    #     })

    # pltr_atoms = pl.DataFrame(data_atoms).unpivot(index="threshold")

    # alt.Chart(pltr_atoms).mark_line().encode(x="threshold", y="value", color="variable").properties(width=800)
    return


@app.cell
def __(Matrix, mo, np):
    mat = mo.ui.anywidget(
        Matrix(matrix=np.eye(2)*10, rows=2, cols=2, max_value=100, min_value=0)
    )
    return (mat,)


@app.cell
def __(mat):
    mat
    return


@app.cell
def __(mat, np):
    out = np.array(mat.matrix)
    # tn, fp, fn, tp = out.ravel()
    return (out,)


@app.cell
def __(y_pred, y_true):
    from sklearn.metrics import classification_report

    classification_report(y_true, y_pred)
    return (classification_report,)


@app.cell
def __(confusion_matrix, fn, fp, tn, tp):
    y_true, y_pred = [], []

    y_true += [1 for i in range(tp)]
    y_pred += [1 for i in range(tp)]
    y_true += [0 for i in range(fp)]
    y_pred += [1 for i in range(fp)]
    y_true += [1 for i in range(fn)]
    y_pred += [0 for i in range(fn)]
    y_true += [0 for i in range(tn)]
    y_pred += [0 for i in range(tn)]

    confusion_matrix(y_true, y_pred)
    return y_pred, y_true


@app.cell
def __():
    from sklearn.metrics import confusion_matrix
    return (confusion_matrix,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
