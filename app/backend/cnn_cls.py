import configparser

import getTrainTest
import numpy as np
import tensorflow as tf
from osgeo import gdal
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import tensorflow.contrib.slim as slim

from sklearn.metrics import confusion_matrix, cohen_kappa_score

COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def write_geotiff(fname, data, N_CLASSES, classes, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    blankcalss = [''] + classes
    ct = gdal.ColorTable()
    for pixel_value in range(N_CLASSES + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))

    band.SetColorTable(ct)
    band.SetCategoryNames(blankcalss)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(N_CLASSES),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1,
                            output_fname='', dataset_format='MEM'):
    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). Return a gdal.Dataset.
    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]
    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """
    Rasterize, in a single image, all the vectors in the given directory.
    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.
    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1

        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray()

        labeled_pixels += a
        ds = None
    return labeled_pixels


if __name__ == "__main__":

    """Hyperparameters"""
    # num_filt_1 = 64 #16     #Number of filters in first conv layer
    # num_filt_2 = 56  #14      #Number of filters in second conv layer
    # num_filt_3 = 32   #8      #Number of filters in third conv layer
    # num_fc_1 =   160  #40       #Number of neurons in hully connected layer
    # max_iterations = 200000
    # batch_size = 64
    # dropout = .8      #Dropout rate in the fully connected layer
    # plot_row = 5        #How many rows do you want to plot in the visualization
    # learning_rate = 2e-3
    # input_norm = False   # Do you want z-score input normalization?

    config = configparser.ConfigParser()
    config.read('/content/drive/MyDrive/000/03project/data/HighResData/calcification.config')

    num_filt_1 = int(config['cnn']['num_filt_1'])
    num_filt_2 = int(config['cnn']['num_filt_2'])
    num_filt_3 = int(config['cnn']['num_filt_3'])
    num_fc_1 = int(config['cnn']['num_fc_1'])
    max_iterations = int(config['cnn']['max_iterations'])
    batch_size = int(config['cnn']['batch_size'])
    dropout = float(config['cnn']['batch_size'])
    plot_row = int(config['cnn']['plot_row'])

    learning_rate = float(config['cnn']['learning_rate'])
    input_norm = bool(config['cnn']['input_norm'])

    method = "CNN"
    # output_rname         = "CNN"

    WorkRout = config['general']['WorkRout']
    image_name = config['general']['image_name']

    root_path = config['general']['WorkRout'] + config['general']['rasterpath']
    train_data_path = config['general']['inputShapefilesPath'] + config['general']['trainName']
    validation_data_path = config['general']['inputShapefilesPath'] + config['general']['testName']

    output_path = WorkRout + config['general']['outputpath'] + method + "_" + image_name + ".tif"
    output_path_txt = WorkRout + config['general']['outputpath'] + method + "_" + image_name + ".txt"

    raster_data_path = root_path + image_name

    # if not os.path.exists(output_path):
    #    os.makedirs(output_path)
    # onlyfiles = [f for f in listdir(output_path) if isfile(join(output_path, f))]
    # output_rname         =    str(len(onlyfiles)+1 )

    training_samples, training_labels, classes, testing_samples, testing_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj = getTrainTest.gettraingtest(
        raster_data_path, train_data_path, validation_data_path)

    # flat_pixels = bands_data.reshape((n_samples, n_bands))
    Number_Class = len(classes)
    # import pdb; pdb.set_trace()
    # X_train = training_samples
    # X_test , X_val =  np.split(testing_samples,2)
    # X_train = training_samples

    X_train, X_val, y_train, y_val = train_test_split(training_samples, training_labels, test_size=0.5,
                                                      random_state=666)

    # X_val   = copy.deepcopy(training_samples)
    X_test = testing_samples

    N = X_train.shape[0]
    Ntest = X_test.shape[0]
    D = X_train.shape[1]

    # y_train = training_labels
    # y_test , y_val =  np.split(testing_labels,2)

    # y_train = training_labels
    # y_val   = copy.deepcopy(training_labels)
    y_test = testing_labels

    print('We have %s observations with %s dimensions' % (N, D))
    # Organize the classes
    num_classes = len(np.unique(y_train))
    base = np.min(y_train)  # Check if data is 0-based
    if base != 0:
        y_train -= base
        y_val -= base
        y_test -= base

    if input_norm:
        mean = np.mean(X_train, axis=0, dtype="uint16")
        variance = np.var(X_train, axis=0)
        X_train -= mean
        flat_pixels -= mean

        # The 1e-9 avoids dividing by zero
        X_train /= np.sqrt(variance) + 1e-9
        X_val -= mean
        X_val /= np.sqrt(variance) + 1e-9
        X_test -= mean
        X_test /= np.sqrt(variance) + 1e-9
        flat_pixels /= np.sqrt(variance) + 1e-9

    # import pdb; pdb.set_trace()
    # Check for the input sizes
    # assert (N>X_train.shape[1]), 'You are feeding a fat matrix for training, are you sure?'
    # assert (Ntest>X_test.shape[1]), 'You are feeding a fat matrix for testing, are you sure?'

    # Proclaim the epochs
    epochs = np.floor(batch_size * max_iterations / N)
    print('Train with approximately %d epochs' % (epochs))

    # Nodes for the input variables
    x = tf.placeholder("float", shape=[None, D], name='Input_data')

    y_ = tf.placeholder(tf.int64, shape=[None], name='Ground_truth')
    keep_prob = tf.placeholder("float")
    bn_train = tf.placeholder(tf.bool)  # Boolean value to guide batchnorm


    # Define functions for initializing variables and standard layers
    # For now, this seems superfluous, but in extending the code
    # to many more layers, this will keep our code
    # read-able

    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    with tf.name_scope("Reshaping_data") as scope:
        x_image = tf.reshape(x, [-1, D, 1, 1])

    initializer = tf.contrib.layers.xavier_initializer()
    """Build the graph"""
    # ewma is the decay for which we update the moving average of the
    # mean and variance in the batch-norm layers
    with tf.name_scope("Conv1") as scope:
        W_conv1 = tf.get_variable("Conv_Layer_1", shape=[5, 1, 1, num_filt_1], initializer=initializer)
        b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
        a_conv1 = conv2d(x_image, W_conv1) + b_conv1

    with tf.name_scope('Batch_norm_conv1') as scope:
        # a_conv1 = tf.contrib.layers.batch_norm(a_conv1,is_training=bn_train,updates_collections=None)
        h_conv1 = tf.nn.relu(a_conv1)

    with tf.name_scope("Conv2") as scope:
        W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4, 1, num_filt_1, num_filt_2], initializer=initializer)
        b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
        a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

    with tf.name_scope('Batch_norm_conv2') as scope:
        # a_conv2 = tf.contrib.layers.batch_norm(a_conv2,is_training=bn_train,updates_collections=None)
        h_conv2 = tf.nn.relu(a_conv2)

    with tf.name_scope("Fully_Connected1") as scope:
        W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D * num_filt_2, num_fc_1], initializer=initializer)
        b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
        h_conv3_flat = tf.reshape(h_conv2, [-1, D * num_filt_2])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    with tf.name_scope("Fully_Connected2") as scope:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_classes], initializer=initializer)
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b_fc2')
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope("SoftMax") as scope:
        # y_result = tf.nn.softmax(tf.matmul(x,h_fc2) + b_fc2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2, labels=y_)
        cost = tf.reduce_sum(loss) / batch_size
        #    cost += regularization*regularizers
        loss_summ = tf.summary.scalar("cross entropy_loss", cost)
    with tf.name_scope("train") as scope:
        tvars = tf.trainable_variables()
        # We clip the gradients to prevent explosion
        grads = tf.gradients(cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = list(zip(grads, tvars))
        train_step = optimizer.apply_gradients(gradients)
        numel = tf.constant([[0]])
        for gradient, variable in gradients:
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            numel += tf.reduce_sum(tf.size(variable))

            h1 = tf.summary.histogram(variable.name, variable)
            h2 = tf.summary.histogram(variable.name + "/gradients", grad_values)
            h3 = tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
    with tf.name_scope("Evaluating_accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(h_fc2, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

    # Define one op to call all summaries
    merged = tf.summary.merge_all()


    def print_tvars():
        tvars = tf.trainable_variables()
        for variable in tvars:
            print(variable.name)
        return


    print_tvars()

    # For now, we collect performances in a Numpy array.
    # In future releases, I hope TensorBoard allows for more
    # flexibility in plotting
    perf_collect = np.zeros((3, int(np.floor(max_iterations / 100))))
    cost_ma = 0.0
    acc_ma = 0.0

    prediction = tf.argmax(h_fc2, 1)


    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)


    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("./log_tb", sess.graph)
        model_summary()
        # print(tf.summary)

        sess.run(tf.global_variables_initializer())

        step = 0  # Step is a counter for filling the numpy array perf_collect
        for i in range(max_iterations):
            batch_ind = np.random.choice(N, batch_size, replace=True)

            if i == 0:
                # Use this line to check before-and-after test accuracy
                result = sess.run(accuracy, feed_dict={x: X_test, y_: y_test, keep_prob: 1.0, bn_train: False})
                acc_test_before = result
            if i % 200 == 0:
                # Check training performance
                result = sess.run([cost, accuracy],
                                  feed_dict={x: X_train, y_: y_train, keep_prob: 1.0, bn_train: False})
                perf_collect[1, step] = acc_train = result[1]
                cost_train = result[0]

                # Check validation performance
                result = sess.run([accuracy, cost, merged],
                                  feed_dict={x: X_val, y_: y_val, keep_prob: 1.0, bn_train: False})
                perf_collect[0, step] = acc_val = result[0]
                cost_val = result[1]
                if i == 0: cost_ma = cost_train
                if i == 0: acc_ma = acc_train
                cost_ma = 0.8 * cost_ma + 0.2 * cost_train
                acc_ma = 0.8 * acc_ma + 0.2 * acc_train

                # Write information to TensorBoard
                # riter.add_summary(result[2], i)
                # writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
                print("At %5.0f/%5.0f Cost: train%5.3f val%5.3f(%5.3f) Acc: train%5.3f val%5.3f(%5.3f) " % (
                i, max_iterations, cost_train, cost_val, cost_ma, acc_train, acc_val, acc_ma))
                step += 1
            sess.run(train_step,
                     feed_dict={x: X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train: True})
        result = sess.run([accuracy, numel], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0, bn_train: False})
        acc_test = result[0]
        print('The network has %s trainable parameters' % (result[1]))

        # import pdb; pdb.set_trace()
        predict_y = sess.run(prediction, feed_dict={x: X_test, keep_prob: 1.0, bn_train: False})

        class_report = metrics.classification_report(y_test, predict_y)
        conf_matrix = confusion_matrix(y_test, predict_y)
        txt_kappa = str(cohen_kappa_score(y_test, predict_y))

        print(class_report)
        print(conf_matrix)

        f = open(output_path_txt, 'w')
        f.write(str(classes))
        f.write("\nclassification_report\n")
        f.write(class_report)

        f.write("\nConfusion matrix\n")
        f.write(str(conf_matrix))

        f.write("\nkappa\n")
        f.write(txt_kappa)

        f.write("\nperf_collect\n Valid accuracy    train accuracy \n")
        f.write(str(np.transpose(perf_collect)))
        f.close()


        def Parallel_manager(flat_pixels, sess):
            sub_flat_pixels = np.array_split(flat_pixels, 10000)
            result = []
            i = 1
            # queue1 = multiprocessing.Queue()
            for single_sub_flat in tqdm(sub_flat_pixels):
                # import pdb; pdb.set_trace()
                sub_result = sess.run(prediction, feed_dict={x: single_sub_flat, keep_prob: 1.0, bn_train: False})
                result = np.append(result, sub_result)
                # print(i)

                #         p = multiprocessing.Process(target=predict_patch, args=(single_sub_flat, classifier,queue1,i))
                #         p.start()
                i = i + 1
                #     for single_sub_flat in sub_flat_pixels:
            #         result = np.append(result, queue1.get())
            return result


        result_classes = Parallel_manager(flat_pixels, sess)
        classification = result_classes.reshape((rows, cols))

        write_geotiff(output_path, classification + 1, num_classes, classes, geo_transform, proj)
