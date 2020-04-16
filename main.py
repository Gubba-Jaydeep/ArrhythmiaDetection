import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydotplus
import io
import cv2
from matplotlib import style






def load_data(path):
    return pd.read_csv(path)

def define_class_names():
    global classes
    classes = {1: 'Normal', 2: 'Ischemic changes (Coronary Artery Disease)', 3: 'Old Anterior Myocardial Infarction',
               4: 'Old Inferior Myocardial Infarction', 5: 'Sinus tachycardy', 6: 'Sinus bradycardy',
               7: 'Ventricular Premature Contraction (PVC)', 8: 'Supraventricular Premature Contraction',
               9: 'Left bundle branch block', 10: 'Right bundle branch block', 11: '1. degree AtrioVentricular block',
               12: '2. degree AV block', 13: '3. degree AV block', 14: 'Left ventricule hypertrophy',
               15: 'Atrial Fibrillation or Flutter', 16: 'Others'}

def describe_the_data(df):
    print(df.head())
    print(df.describe())


def transform_the_data(df):
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    return df

def check_if_data_has_na(df):
    print(df.isnull())
    print('is Data having missing values:', df.isnull().any().any())
    print('total misssing values: ', df.isnull().sum().sum())
    print('no of columns having missing values: ', df.isnull().any().sum())
    dfNullCount = df.isnull().sum()
    print('\n\nNull frequency count')
    print(dfNullCount[dfNullCount > 0])

def plot_graph_to_show_na_frequency(df):
    fig1 = plt.figure()
    cellText = [[], []]
    columns = []
    N = len(df)

    rows = ['n(NA)', 'n(NA)/N *100']
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    for i in range(len(df.columns) - 1):
        if df[df.columns[i]].isnull().sum() > 0:
            cellText[0].append(df[df.columns[i]].isnull().sum())
            cellText[1].append(round((cellText[0][-1] / N) * 100, 1))
            columns.append(str(df.columns[i]) + '(' + str(i + 1) + ')')

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    y_offset = np.zeros(len(columns))

    n_rows = len(cellText)

    plt.bar(index, [N] * len(cellText[0]), bar_width, bottom=y_offset, color=colors[1])
    plt.bar(index, cellText[0], bar_width, bottom=y_offset, color='#F9927C')

    plt.table(cellText=cellText, rowLabels=rows, rowColours=colors, colLabels=columns, loc='bottom')
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.xticks([])
    plt.title('Frequency of NA values')
    plt.show()


def get_the_shape_of_data_after_removing_all_na(df):
    print('shape of data: ', df.shape)
    df_drop_all_na = df.dropna()
    print('shape after droping all nas: ', df_drop_all_na.shape)

def plot_class_wise_frequency_classification(df):
    df = pd.read_csv('/Users/jaydeep/MachineLearning/MajorProjectEDG/arrhythmia.data.csv')

    def get_freq(d):
        l = []
        for i in range(1, 17):
            if i in d:
                l.append(d[i])
            else:
                l.append(0)
        return l

    df[df.columns[13]]=df[df.columns[13]].replace('?', np.nan)
    freq_if_col14_removed = get_freq(df.dropna()['Class'].value_counts().sort_index())
    freq_imputed = get_freq(df.dropna(axis=1, how='any', thresh=400)['Class'].value_counts().sort_index())
    df.replace('?', np.nan, inplace=True)
    freq_after_removingna = get_freq(df.dropna(axis=1, how='any', thresh=400).dropna()['Class'].value_counts().sort_index())
    labels = list(map(str, range(1, 17)))
    x = np.arange(len(labels))  # the label locations
    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, freq_imputed, width, label='imputed ~30 na after removing feature 14')
    rects2 = ax.bar(x, freq_if_col14_removed, width, label='feature 14 na patients removed')
    rects3 = ax.bar(x + width, freq_after_removingna, width, label='remove ~30 na patients after removing feature 14')

    ax.set_ylabel('Frequency')
    ax.set_title('Historam of Classification Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

def remove_the_column_not_having_atlease_440_nonNA_values(df):
    df.dropna(axis=1, how='any', thresh=400, inplace=True)
    return df
def impute_the_data_with_median(df):
    df = df.fillna(df.median())
    return df
def split_data(df):
    x = df.values[:, :-1]
    y = df.values[:, -1]
    from sklearn.model_selection import train_test_split
    global x_train_50, x_test_50, y_train_50, y_test_50
    global x_train_30, x_test_70, y_train_30, y_test_70
    global x_train_70, x_test_30, y_train_70, y_test_30
    global x_train_100, x_test_100, y_train_100, y_test_100
    x_train_50, x_test_50, y_train_50, y_test_50 = train_test_split(x,y,test_size=0.5)
    x_train_30, x_test_70, y_train_30, y_test_70 = train_test_split(x, y, test_size=0.7)
    x_train_70, x_test_30, y_train_70, y_test_30 = train_test_split(x, y, test_size=0.3)
    x_train_100, x_test_100, y_train_100, y_test_100 = x,x,y,y

def imputed_flow(df):
    split_data(df)

    clf_gini = DecisionTreeClassifier(criterion="gini", splitter='best', max_leaf_nodes=16, ccp_alpha=0.0001)



    clf_nb = GaussianNB()
    clf_svm = SVC(C=10000, kernel="rbf")

    global cart_gini_accuracies_imputed
    cart_gini_accuracies_imputed= []
    global NB_accuracies_imputed
    NB_accuracies_imputed = []
    global svm_accuracies_imputed
    svm_accuracies_imputed = []

    fit_and_cal_accuracy(clf_gini, x_train_50, x_test_50, y_train_50, y_test_50, cart_gini_accuracies_imputed)
    fit_and_cal_accuracy(clf_nb, x_train_50, x_test_50, y_train_50, y_test_50, NB_accuracies_imputed)
    fit_and_cal_accuracy(clf_svm, x_train_50, x_test_50, y_train_50, y_test_50, svm_accuracies_imputed)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_imputed[-1])

    fit_and_cal_accuracy(clf_gini, x_train_30, x_test_70, y_train_30, y_test_70, cart_gini_accuracies_imputed)
    fit_and_cal_accuracy(clf_nb, x_train_30, x_test_70, y_train_30, y_test_70, NB_accuracies_imputed)
    fit_and_cal_accuracy(clf_svm, x_train_30, x_test_70, y_train_30, y_test_70, svm_accuracies_imputed)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_imputed[-1])

    fit_and_cal_accuracy(clf_gini, x_train_70, x_test_30, y_train_70, y_test_30, cart_gini_accuracies_imputed)
    fit_and_cal_accuracy(clf_nb, x_train_70, x_test_30, y_train_70, y_test_30, NB_accuracies_imputed)
    fit_and_cal_accuracy(clf_svm, x_train_70, x_test_30, y_train_70, y_test_30, svm_accuracies_imputed)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_imputed[-1])

    fit_and_cal_accuracy(clf_gini, x_train_100, x_test_100, y_train_100, y_test_100, cart_gini_accuracies_imputed)
    fit_and_cal_accuracy(clf_nb, x_train_100, x_test_100, y_train_100, y_test_100, NB_accuracies_imputed)
    fit_and_cal_accuracy(clf_svm, x_train_100, x_test_100, y_train_100, y_test_100, svm_accuracies_imputed)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_imputed[-1])





def featureSelection_flow(df):
    clf_gini = DecisionTreeClassifier(criterion="gini", splitter='best', max_leaf_nodes=16)
    clf_gini.fit(x_train_100, y_train_100)
    importantFeatures = clf_gini.feature_importances_
    importantFeatures = df.columns[:-1][importantFeatures>0]
    split_data(df[list(importantFeatures)+['Class']])
    clf_gini = DecisionTreeClassifier(criterion="gini", splitter='best', max_leaf_nodes=16)
    clf_nb = GaussianNB()
    clf_svm = SVC(C=10000, kernel="rbf")

    global cart_gini_accuracies_featureSelection
    cart_gini_accuracies_featureSelection = []
    global NB_accuracies_featureSelection
    NB_accuracies_featureSelection = []
    global svm_accuracies_featureSelection
    svm_accuracies_featureSelection = []

    fit_and_cal_accuracy(clf_gini, x_train_50, x_test_50, y_train_50, y_test_50, cart_gini_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_nb, x_train_50, x_test_50, y_train_50, y_test_50, NB_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_svm, x_train_50, x_test_50, y_train_50, y_test_50, svm_accuracies_featureSelection)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_featureSelection[-1])

    fit_and_cal_accuracy(clf_gini, x_train_30, x_test_70, y_train_30, y_test_70, cart_gini_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_nb, x_train_30, x_test_70, y_train_30, y_test_70, NB_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_svm, x_train_30, x_test_70, y_train_30, y_test_70, svm_accuracies_featureSelection)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_featureSelection[-1])

    fit_and_cal_accuracy(clf_gini, x_train_70, x_test_30, y_train_70, y_test_30, cart_gini_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_nb, x_train_70, x_test_30, y_train_70, y_test_30, NB_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_svm, x_train_70, x_test_30, y_train_70, y_test_30, svm_accuracies_featureSelection)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_featureSelection[-1])

    fit_and_cal_accuracy(clf_gini, x_train_100, x_test_100, y_train_100, y_test_100, cart_gini_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_nb, x_train_100, x_test_100, y_train_100, y_test_100, NB_accuracies_featureSelection)
    fit_and_cal_accuracy(clf_svm, x_train_100, x_test_100, y_train_100, y_test_100, svm_accuracies_featureSelection)
    store_best_cart_classifer(clf_gini, cart_gini_accuracies_featureSelection[-1])


def fit_and_cal_accuracy(classifer, x_train, x_test, y_train, y_test, accuracies,):
    classifer.fit(x_train, y_train)
    y_pred = classifer.predict(x_test)
    accuracy= accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)



def plot_decision_tree(tree,path,df,classes):
    f=io.StringIO()
    export_graphviz(tree, out_file=f, filled=True, rounded=True, special_characters=True, feature_names=list(df.columns)[:-1],class_names=list(classes.values()))
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=cv2.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)

def initialize_suite_variables():
    global best_cart_accuracy, best_cart_classifier
    best_cart_accuracy=0
    best_cart_classifier=None
def store_best_cart_classifer(classifer, accuracy):
    global best_cart_accuracy, best_cart_classifier
    if accuracy > best_cart_accuracy:
        best_cart_accuracy = accuracy
        best_cart_classifier=classifer

def plot_accuracies_table():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    table_data = [
        [str(round(NB_accuracies_imputed[0], 2)) + ' | ' + str(
            round(NB_accuracies_imputed[1], 2)) + '\n------------\n' + str(
            round(NB_accuracies_imputed[2], 2)) + ' | ' + str(round(NB_accuracies_imputed[3], 2)),
         str(round(NB_accuracies_featureSelection[0], 2)) + ' | ' + str(
             round(NB_accuracies_featureSelection[1], 2)) + '\n------------\n' + str(
             round(NB_accuracies_featureSelection[2], 2)) + ' | ' + str(round(NB_accuracies_featureSelection[3], 2))],
        [str(round(svm_accuracies_imputed[0], 2)) + ' | ' + str(
            round(svm_accuracies_imputed[1], 2)) + '\n------------\n' + str(
            round(svm_accuracies_imputed[2], 2)) + ' | ' + str(round(svm_accuracies_imputed[3], 2)),
         str(round(svm_accuracies_featureSelection[0], 2)) + ' | ' + str(
             round(svm_accuracies_featureSelection[1], 2)) + '\n------------\n' + str(
             round(svm_accuracies_featureSelection[2], 2)) + ' | ' + str(round(svm_accuracies_featureSelection[3], 2))],
        [str(round(cart_gini_accuracies_imputed[0], 2)) + ' | ' + str(
            round(cart_gini_accuracies_imputed[1], 2)) + '\n------------\n' + str(
            round(cart_gini_accuracies_imputed[2], 2)) + ' | ' + str(round(cart_gini_accuracies_imputed[3], 2)),
         str(round(cart_gini_accuracies_featureSelection[0], 2)) + ' | ' + str(
             round(cart_gini_accuracies_featureSelection[1], 2)) + '\n------------\n' + str(
             round(cart_gini_accuracies_featureSelection[2], 2)) + ' | ' + str(round(cart_gini_accuracies_featureSelection[3], 2))],

    ]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(table_data)))
    table = ax.table(cellText=table_data, rowLabels=['Naive Bayes', 'SVM', 'CART'], rowColours=colors,
                     colLabels=['Imputed V14', 'Feature Selection'], loc='center',rowLoc='center')
    table.set_fontsize(8)
    table.scale(1, 6)
    ax.axis('off')
    plt.show()
if __name__=="__main__":

    initialize_suite_variables()
    df=load_data('arrhythmia.data.csv')
    define_class_names()
    describe_the_data(df)
    transform_the_data(df)
    check_if_data_has_na(df)
    plot_graph_to_show_na_frequency(df)
    get_the_shape_of_data_after_removing_all_na(df)
    plot_class_wise_frequency_classification(df)
    df=remove_the_column_not_having_atlease_440_nonNA_values(df)
    df=impute_the_data_with_median(df)
    imputed_flow(df)
    featureSelection_flow(df)
    plot_decision_tree(best_cart_classifier, 'dec_treeP.png', df, classes)
    plot_accuracies_table()







