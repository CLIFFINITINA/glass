path = '../input/glass/glass.csv'
df = pd.read_csv(path)
df.head()
features = df.drop('Type', axis=1)
labels = df['Type'].values
labels = labels.reshape(len(labels), 1)
features.describe().T
scaler = StandardScaler()
features = scaler.fit_transform(features)
dataset = np.append(features, labels, axis=1).tolist()
def kfold(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
        
    return dataset_split
n_folds = 5

folds = kfold(dataset, n_folds)
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
def random_subset(train):
    n_records = len(train)
    n_features = len(train[0])
    subsets = [train[randrange(n_records)][i] for i in range(n_features)]
    return subsets
def best_match(subsets, test_row):
    distances = list()
    
    for subset in subsets:
        dist = euclidean_distance(subset, test_row)
        distances.append((subset, dist))
        
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]
def lvq(train_set, n_subsets, lrate, epochs):
    subsets = [random_subset(train_set) for i in range(n_subsets)]

    for epoch in range(epochs):
        rate = lrate * (1.0-(epoch/float(epochs)))
        
        for row in train_set:
            bmu = best_match(subsets, row)
            
            for i in range(len(row)-1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
                    
    return subsets
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
def train_test_split(folds, fold):
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = list()
    return train_set, test_set
lrate = 0.1
epochs = 500
n_subsets = 50
scores = list()

for fold in folds:
    train_set, test_set = train_test_split(folds, fold)

    for row in fold:
        test_set.append(list(row))

    subsets = lvq(train_set, n_subsets, lrate, epochs)
    y_hat = list()
    
    for test_row in test_set:
        output = best_match(subsets, test_row)[-1]
        y_hat.append(output)
    
    y = [row[-1] for row in fold]
    scores.append(accuracy(y, y_hat))
print('Accuracy per fold: {:}'.format(scores))
print('Max Accuracy: {:.3f}'.format(max(scores)))