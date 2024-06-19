from detection import SVMClassifier


def main():
    model_hog = SVMClassifier(
        feature_type="hog",
        save_pickle_path="checkpoint/svm.pkl"
    )
    model_hog.train()
    model_color = SVMClassifier(
        feature_type="color",
        save_pickle_path="checkpoint/color.pkl"
    )
    model_color.train()
    model_lbp = SVMClassifier(
        feature_type="lbp",
        save_pickle_path="checkpoint/lbp.pkl"
    )
    model_lbp.train()
    model_resize = SVMClassifier(
        feature_type="resize",
        save_pickle_path="checkpoint/resize.pkl"
    )
    model_resize.train()
    

if __name__ == "__main__":
    main()