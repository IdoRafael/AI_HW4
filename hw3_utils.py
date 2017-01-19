data_path = r'data/arrhythmia.data'


def to_int(l):
    res = []
    for w in l:
        try:
            res.append(float(w))
        except:
            res.append(w)

    return res


def parse_data(path):
    res = []
    with open(path) as f:
        for l in f:
            res.append(to_int(l.split(',')))
    return res


def create_dataset(path=r'data/ecg_examples.data'):
    '''
    return the dataset that will be used in HW 3
    prameters:
    :param path: the path of the csv data file (default value is data/ecg_examples.data)

    :returns: the tuple feature, label
    features - a list of lists where the ith list is the feature vector of patient i. the last value in each list is the label of the current patient
    '''
    features = parse_data(path)
    # label = [x.pop() for x in features]
    return features


def create_test_set(path=r'data/ecg_bonus.data'):
    '''
    return the dataset that will be used in HW 3 test part
    prameters:
    :param path: - the path of the csv data file (default value is data/ecg_bonus.data)

    :returns: the tuple feature, label
    features - a list of lists where the ith list is the feature vector of patient i
    '''
    return parse_data(path)


def write_prediction(pred, path='results.data'):
    '''
    write the prediction of the test set into a file for submission
    prameters:
    :param pred: - a list of result the ith entry represent the ith subject (as integers of 1 or 0, where 1 is a healthy patient and 0 otherwise)
    :param path: - the path of the csv data file will be saved to(default value is res.data)

    :return: None
    '''
    output = []
    for l in pred:
        output.append(l)
    with open(path, 'w') as f:
        f.write(', '.join([str(x) for x in output]) + '\n')


class abstract_classifier_factory:
    '''
    an abstruct class for classifier factory
    '''
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''
        raise Exception('Not implemented')


class abstract_classifier:
    '''
        an abstruct class for classifier
    '''

    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''
        raise Exception('Not implemented')
