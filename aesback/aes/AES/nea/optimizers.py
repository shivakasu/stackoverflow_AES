import keras.optimizers as opt
from .config import ModelConfig as MC

def get_optimizer():

	clipvalue = 0
	clipnorm = 10

	if MC.OPTIMIZER == 'rmsprop':
		optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'sgd':
		optimizer = opt.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'adagrad':
		optimizer = opt.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'adadelta':
		optimizer = opt.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'adam':
		optimizer = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'amsgrad':
		optimizer = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue, amsgrad=True)
	elif MC.OPTIMIZER == 'adamax':
		optimizer = opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
	elif MC.OPTIMIZER == 'amsgrad':
		optimizer = opt.AMSGrad(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)

	return optimizer
