from __future__ import absolute_import
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error

class Adversarial_Training(Model):
    _at_loss = None
    # set up
    def setup_at_loss(self, loss_func=mean_squared_error, eps=0.25/255.0, alpha=1.0):
        self._loss_func = loss_func
        self._alpha = alpha
        self._at_loss = self.at_loss(eps)
        return self
    # loss
    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self._at_loss is not None:
            losses += [ self._alpha * self._at_loss ]
        return losses
    # VAT loss
    def at_loss(self, eps):
        # original loss
        loss_orig = self._loss_func(self.inputs[-1], self.outputs[0])
        # gradients
        grads = K.stop_gradient(K.gradients(loss_orig, self.inputs[:-1]))[0]
        # perterbed samples
        new_inputs = self.inputs[:-1] + eps * K.sign(grads)
        # estimation for the perturbated samples
        outputs_perturb = self.call([new_inputs, self.inputs[-1]])
        # computing losses
        loss = self._loss_func(self.inputs[-1], outputs_perturb)
        return loss