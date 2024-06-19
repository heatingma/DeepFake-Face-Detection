from detection import SVMClassifier


def test_svm_hog():
    model = SVMClassifier(
        feature_type="hog",
        pretrain=True,
        pretrained_pickle_path="checkpoint/svm_hog.pkl"
    )
    acc, f1, auc = model.test()
    message = f"SVM(HOG) Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)


def test_svm_colcor():
    model = SVMClassifier(
        feature_type="color",
        pretrain=True,
        pretrained_pickle_path="checkpoint/svm_color.pkl"
    )
    acc, f1, auc = model.test()
    message = f"SVM(Color) Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    

def test_svm_lbp():
    model = SVMClassifier(
        feature_type="lbp",
        pretrain=True,
        pretrained_pickle_path="checkpoint/svm_lbp.pkl"
    )
    acc, f1, auc = model.test()
    message = f"SVM(LBP) Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    
    
def test_svm_resize():
    model = SVMClassifier(
        feature_type="resize",
        pretrain=True,
        pretrained_pickle_path="checkpoint/svm_resize.pkl"
    )
    acc, f1, auc = model.test()
    message = f"SVM(Resize) Accuracy: {acc}%, F1: {f1}%, AUC: {auc}%"
    print(message)
    

if __name__ == "__main__":
    test_svm_hog()
    test_svm_colcor()
    test_svm_lbp()
    test_svm_resize()