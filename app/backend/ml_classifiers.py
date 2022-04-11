from pathlib import Path

from backend import getTrainTest
from backend import writetif
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MLClassifier:
    CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'RF': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'SVM': SVC(class_weight='balanced', C=1000.0, gamma=0.00000001),
        # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        'MLP': MLPClassifier(activation='relu',
                             alpha=1e-4,
                             batch_size='auto',
                             beta_1=0.9,
                             beta_2=0.999,
                             early_stopping=False,
                             epsilon=1e-08,
                             hidden_layer_sizes=(9,),
                             learning_rate='constant',
                             learning_rate_init=.1,
                             max_iter=200,
                             momentum=0.9,
                             nesterovs_momentum=True,
                             power_t=0.5,
                             random_state=1,
                             shuffle=True,
                             solver='adam',
                             tol=0.0001,
                             validation_fraction=0.1,
                             verbose=200,
                             warm_start=True),
        'KNN': KNeighborsClassifier(5),
        'gp': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        'dt': DecisionTreeClassifier(max_depth=5),
        'RF2': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        'mlp2': MLPClassifier(alpha=1),
        'ap': AdaBoostClassifier(),
        'qd': QuadraticDiscriminantAnalysis()
    }

    def svm(self, c, g):
        classifier = SVC(class_weight='balanced', C=c, gamma=g, verbose=True)
        classifier.fit(self.training_samples, self.training_labels)
        predicted_labels = classifier.predict(self.testing_samples)

        #
        # Validate the results
        #
        res = "\n##############################################\n"
        res = res + "\n" + str(self.CLASSIFIERS[self.method])
        target_names = ['Class %s' % s for s in self.classes]
        res = res + "\n" + str("target_names:\n%s" %
                               target_names)
        aa = metrics.confusion_matrix(self.verification_labels, predicted_labels)
        res = res + "\n" + str("Confussion matrix:\n%s" %
                               metrics.confusion_matrix(self.verification_labels, predicted_labels))
        res = res + "\n" + str("Classification report:\n%s" %
                               metrics.classification_report(self.verification_labels, predicted_labels,
                                                             target_names=target_names))
        q = metrics.accuracy_score(self.verification_labels, predicted_labels)
        res = res + "\n" + str("Classification accuracy: %f" %
                               metrics.accuracy_score(self.verification_labels, predicted_labels))
        f = open(self.output_path_txt, 'w')
        f.write(str(self.classes))
        f.write("\nclassification_report\n")
        f.write(res)
        f.close()
        return classifier, res, self.classes, self.raster_data_path, self.output_path

    def rf(self, j, s):
        classifier = RandomForestClassifier(n_jobs=j, n_estimators=s, class_weight='balanced')
        classifier.fit(self.training_samples, self.training_labels)
        predicted_labels = classifier.predict(self.testing_samples)

        #
        # Validate the results
        #
        res = "\n##############################################\n"

        res = res + "\n" + str(self.CLASSIFIERS[self.method])

        target_names = ['Class %s' % s for s in self.classes]

        res = res + "\n" + str("target_names:\n%s" %
                               target_names)
        aa = metrics.confusion_matrix(self.verification_labels, predicted_labels)

        res = res + "\n" + str("Confussion matrix:\n%s" %
                               metrics.confusion_matrix(self.verification_labels, predicted_labels))

        res = res + "\n" + str("Classification report:\n%s" %
                               metrics.classification_report(self.verification_labels, predicted_labels,
                                                             target_names=target_names))

        q = metrics.accuracy_score(self.verification_labels, predicted_labels)
        res = res + "\n" + str("Classification accuracy: %f" %
                               metrics.accuracy_score(self.verification_labels, predicted_labels))

        f = open(self.output_path_txt, 'w')
        f.write(str(self.classes))
        f.write("\nclassification_report\n")
        f.write(res)
        f.close()

        return classifier, res, self.classes, self.raster_data_path, self.output_path

    def mlp(self, p1, p2, p3, p4):
        classifier = MLPClassifier(hidden_layer_sizes=(p1, p2, p3, p4))
        classifier.fit(self.training_samples, self.training_labels)
        predicted_labels = classifier.predict(self.testing_samples)

        #
        # Validate the results
        #
        res = "\n##############################################\n"

        res = res + "\n" + str(self.CLASSIFIERS[self.method])

        target_names = ['Class %s' % s for s in self.classes]

        res = res + "\n" + str("target_names:\n%s" %
                               target_names)
        aa = metrics.confusion_matrix(self.verification_labels, predicted_labels)

        res = res + "\n" + str("Confussion matrix:\n%s" %
                               metrics.confusion_matrix(self.verification_labels, predicted_labels))

        res = res + "\n" + str("Classification report:\n%s" %
                               metrics.classification_report(self.verification_labels, predicted_labels,
                                                             target_names=target_names))

        q = metrics.accuracy_score(self.verification_labels, predicted_labels)
        res = res + "\n" + str("Classification accuracy: %f" %
                               metrics.accuracy_score(self.verification_labels, predicted_labels))

        f = open(self.output_path_txt, 'w')
        f.write(str(self.classes))
        f.write("\nclassification_report\n")
        f.write(res)
        f.close()

        return classifier, res, self.classes, self.raster_data_path, self.output_path

    def knn(self, knn_param):
        classifier = KNeighborsClassifier(knn_param)
        classifier.fit(self.training_samples, self.training_labels)
        predicted_labels = classifier.predict(self.testing_samples)

        #
        # Validate the results
        #
        res = "\n##############################################\n"

        res = res + "\n" + str(self.CLASSIFIERS[self.method])

        target_names = ['Class %s' % s for s in self.classes]

        res = res + "\n" + str("target_names:\n%s" %
                               target_names)
        aa = metrics.confusion_matrix(self.verification_labels, predicted_labels)

        res = res + "\n" + str("Confussion matrix:\n%s" %
                               metrics.confusion_matrix(self.verification_labels, predicted_labels))

        res = res + "\n" + str("Classification report:\n%s" %
                               metrics.classification_report(self.verification_labels, predicted_labels,
                                                             target_names=target_names))

        q = metrics.accuracy_score(self.verification_labels, predicted_labels)
        res = res + "\n" + str("Classification accuracy: %f" %
                               metrics.accuracy_score(self.verification_labels, predicted_labels))

        f = open(self.output_path_txt, 'w')
        f.write(str(self.classes))
        f.write("\nclassification_report\n")
        f.write(res)
        f.close()

        return classifier, res, self.classes, self.raster_data_path, self.output_path

    def set_svm_parameters(self, classifier_method, input_raster_image, training_data_location, test_data_location,
                           output_data_location, c, gamma):
        self.method = classifier_method
        svmC = int(c)
        svmGama = float(gamma)

        train_data_path = training_data_location
        validation_data_path = test_data_location

        image_name = Path(input_raster_image).stem

        self.output_path = output_data_location + "/" + self.method + "_" + image_name + ".tif"
        self.output_path_txt = output_data_location + "/" + self.method + "_" + image_name + ".txt"

        self.raster_data_path = input_raster_image

        self.training_samples, self.training_labels, self.classes, self.testing_samples, self.verification_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj = getTrainTest.gettraingtest(
            self.raster_data_path, train_data_path, validation_data_path)

        classifier, res, self.classes, self.raster_data_path, self.output_path = self.svm(svmC, svmGama)

        writetif.writegeoresults(self.raster_data_path, classifier, self.classes, self.output_path)

    def set_rf_parameters(self, classifier_method, input_raster_image, training_data_location, test_data_location,
                          output_data_location, n_jobs, n_estimators):
        self.method = classifier_method

        rf_para1_n_jobs = int(n_jobs)
        rf_para2_n_estimators = int(n_estimators)

        train_data_path = training_data_location
        validation_data_path = test_data_location

        image_name = Path(input_raster_image).stem

        self.output_path = output_data_location + "/" + self.method + "_" + image_name + ".tif"
        self.output_path_txt = output_data_location + "/" + self.method + "_" + image_name + ".txt"

        self.raster_data_path = input_raster_image

        self.training_samples, self.training_labels, self.classes, self.testing_samples, self.verification_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj = getTrainTest.gettraingtest(
            self.raster_data_path, train_data_path, validation_data_path)

        classifier, res, self.classes, self.raster_data_path, self.output_path = self.rf(rf_para1_n_jobs,
                                                                                         rf_para2_n_estimators)

        writetif.writegeoresults(self.raster_data_path, classifier, self.classes, self.output_path)

    def set_mlp_parameters(self, classifier_method, input_raster_image, training_data_location, test_data_location,
                           output_data_location, fst_hdn_lyr, snd_hdn_lyr, trd_hdn_lyr, frt_hdn_lyr):
        self.method = classifier_method

        first_hidden_layer = int(fst_hdn_lyr)
        second_hidden_layer = int(snd_hdn_lyr)
        third_hidden_layer = int(trd_hdn_lyr)
        fourth_hidden_layer = int(frt_hdn_lyr)

        train_data_path = training_data_location
        validation_data_path = test_data_location

        image_name = Path(input_raster_image).stem

        self.output_path = output_data_location + "/" + self.method + "_" + image_name + ".tif"
        self.output_path_txt = output_data_location + "/" + self.method + "_" + image_name + ".txt"

        self.raster_data_path = input_raster_image

        self.training_samples, self.training_labels, self.classes, self.testing_samples, self.verification_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj = getTrainTest.gettraingtest(
            self.raster_data_path, train_data_path, validation_data_path)

        classifier, res, self.classes, self.raster_data_path, self.output_path = self.mlp(first_hidden_layer,
                                                                                          second_hidden_layer,
                                                                                          third_hidden_layer,
                                                                                          fourth_hidden_layer)

        writetif.writegeoresults(self.raster_data_path, classifier, self.classes, self.output_path)

    def set_knn_parameters(self, classifier_method, input_raster_image, training_data_location, test_data_location,
                           output_data_location, knn_param):
        self.method = classifier_method
        knn_parameter = int(knn_param)

        train_data_path = training_data_location
        validation_data_path = test_data_location

        image_name = Path(input_raster_image).stem

        self.output_path = output_data_location + "/" + self.method + "_" + image_name + ".tif"
        self.output_path_txt = output_data_location + "/" + self.method + "_" + image_name + ".txt"

        self.raster_data_path = input_raster_image

        self.training_samples, self.training_labels, self.classes, self.testing_samples, self.verification_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj = getTrainTest.gettraingtest(
            self.raster_data_path, train_data_path, validation_data_path)

        classifier, res, self.classes, self.raster_data_path, self.output_path = self.knn(knn_parameter)

        writetif.writegeoresults(self.raster_data_path, classifier, self.classes, self.output_path)
