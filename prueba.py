def plot_cv_roc(model_name):
    n_splits = 5

    cv = StratifiedKFold(n_splits=n_splits)

    classifier = models[model_name]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n{model_name}",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()










    fig, axes = plt.subplots(
        int(n_classes/2), int(n_classes/2), figsize=(13, 12))

    mean_fpr = np.linspace(0, 1, 100)
    all_mean_tprs = []
    all_std_aucs = []

    for i, class_label in enumerate(np.unique(y)):

        tprs = []
        aucs = []

        mean_fpr = np.linspace(0, 1, 100)

        for fold, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            y_proba = classifier.predict_proba(X[test])

            fpr, tpr, _ = roc_curve(y[test] == class_label, y_proba[:, i])

            ax = axes[int(i / 2)][i % 2]
            ax.plot(fpr,
                    tpr,
                    label=f'Fold {fold+1} (AUC = {auc(fpr, tpr):.2f})',
                    alpha=0.3,
                    )

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        all_mean_tprs.append(mean_tpr)
        all_std_aucs.append(std_auc)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color='blue',
            label=f'Mean ROC (AUC = {mean_auc:.2f} ' +
            r'$\pm$ ' + f'{std_auc:.3f})',
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            lw=1,
            color='black',
            label='Chance Level (AUC = 0.5)',
            alpha=0.8
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title=f'Class {class_label}'
        )
        ax.axis('square')
        ax.legend(loc="lower right")

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        plt.figure(2)
        plt.plot(mean_fpr, mean_tprs, label=f'Class {class_label} (AUC = {mean_auc:.2f} ' +
                 r'$\pm$ ' + f'{std_auc:.3f})', alpha=0.3)

    overall_mean_tpr = np.mean(all_mean_tprs, axis=0)
    overall_mean_auc = auc(mean_fpr, overall_mean_tpr)
    overall_std_auc = np.std(all_std_aucs)
