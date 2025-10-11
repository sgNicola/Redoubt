parser = argparse.ArgumentParser()
parser.add_argument("--postprocessor", type=str, default='msp') # 'msp' 'ebo' 'maxlogit' 'Mahalanobis' 'ash' 'react' 'knn' 'gen' 'vim'
args = parser.parse_args()

def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

