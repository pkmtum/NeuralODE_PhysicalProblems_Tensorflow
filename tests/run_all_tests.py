import unittest

import tensorflow as tf
from problems import *
from api_tests import *
from gradient_tests import *
from odeint_tests import *
from model_tests import *

if __name__ == '__main__':
    with tf.device('cpu'):
        unittest.main()
