# A2S (AdaFusion) – Curvature-Aware Adam–SGD Hybrid Optimizer

This repo contains a simple NumPy implementation of a feed-forward neural network trained on MNIST, plus a custom optimizer called **A2S** (a.k.a. **AdaFusion**) that blends **Adam** and **SGD with momentum** using a curvature-aware schedule.

---

## 1. What is A2S / AdaFusion?

A2S starts training like Adam (fast, adaptive updates) and gradually blends towards SGD with momentum (more stable, less “over-adaptive”) as training progresses.

Key ideas:

- **Adaptive phase (early):** behave mostly like Adam to make fast progress.
- **Curvature-aware scaling:** if gradients change a lot between steps (high curvature), shrink the effective learning rate.
- **Blending phase:** slowly reduce Adam’s weight and increase SGD’s contribution.
- **SGD phase (late):** in principle, end up closer to a “classic” SGD+momentum regime for flatter minima.

The optimizer is implemented as:

```python
class A2S:
   
    def __init__(self,
                 lr_adam=0.001,
                 lr_sgd=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 momentum=0.9,
                 curvature_k=0.05,
                 switch_iter=1500,
                 blend_iters=3000,
                 epsilon=1e-8,
                 window_size=10):

        self.lr_adam = lr_adam
        self.lr_sgd = lr_sgd
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.curvature_k = curvature_k
        self.switch_iter = switch_iter
        self.blend_iters = blend_iters
        self.epsilon = epsilon
        self.window_size = window_size

        self.t = 0
        self.m = {}
        self.v = {}
        self.buf = {}
        self.prev_grad = None
        self.grad_history = []

    def update(self, params, grads):
        self.t += 1
        t = self.t

        # recent gradient history
        self.grad_history.append({k: grads[k].copy() for k in grads})
        if len(self.grad_history) > self.window_size:
            self.grad_history.pop(0)

        flat = np.concatenate([g[k].flatten() for g in self.grad_history for k in g])
        grad_variance = np.var(flat) + 1e-12  # currently not used, kept for extension

        # curvature estimate from gradient difference
        if self.prev_grad is None:
            curvature = 0
        else:
            curvature = np.sqrt(sum(
                np.sum((grads[k] - self.prev_grad[k])**2)
                for k in grads
            ))
        self.prev_grad = {k: grads[k].copy() for k in grads}

        # curvature-based LR scaling
        lr_scale = 1.0 / (1.0 + self.curvature_k * curvature)
        lr_adam_eff = self.lr_adam * lr_scale
        lr_sgd_eff = self.lr_sgd * lr_scale

        # init state
        if not self.m:
            for k in params:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
                self.buf[k] = np.zeros_like(params[k])

        # blended Adam + SGD-momentum step
        for k in params:
            g = grads[k]

            # Adam moments
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*(g*g)

            m_hat = self.m[k] / (1-self.beta1**t)
            v_hat = self.v[k] / (1-self.beta2**t)

            adam_step = lr_adam_eff * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # SGD momentum buffer
            self.buf[k] = self.momentum*self.buf[k] + g
            sgd_step = lr_sgd_eff * self.buf[k]

            # blend weight: w=1 → pure Adam, w=0 → pure SGD
            if t <= self.switch_iter:
                w = 1.0
            elif t >= self.switch_iter + self.blend_iters:
                w = 0.0
            else:
                w = 1 - (t - self.switch_iter) / self.blend_iters

            params[k] -= w * adam_step + (1-w) * sgd_step


#Meaning of the Hyperparameters

lr_adam: base learning rate for the Adam component.

lr_sgd: base learning rate for the SGD+momentum component.

beta1, beta2: Adam’s first and second moment decay rates.

momentum: momentum factor for the SGD part.

curvature_k: controls how aggressively curvature shrinks the learning rate.

Higher curvature_k → more shrinkage when gradients change a lot.

switch_iter: iteration where we start blending away from pure Adam.

blend_iters: number of iterations over which we move from Adam to SGD.

epsilon: small constant for numerical stability.

window_size: how many recent gradient snapshots we keep (for variance; currently not used in the update, but easy to plug in later for variance-based logic).

# Model & Training Setup (MNIST)

The training script uses:

Model: 3-layer fully connected network

784 → 128 → 64 → 10 with ReLU activations and softmax output.

Dataset: MNIST via fetch_openml("mnist_784").

Loss: cross-entropy.

Batch size: 64.

Epochs: 20.

Baselines: Adam, RMSProp, SGD+Momentum.

Each optimizer is plugged into the same training loop via a simple update(params, grads) interface.

# Why Mix Adam and SGD?

Intended behaviour (in theory):

Early: Adam handles noisy, unscaled gradients well and converges quickly.

Later: SGD+momentum is less “over-adaptive”, can help escape sharp minima and land in flatter regions.

A2S tries to enjoy fast start (Adam) + nicer end behaviour (SGD) while avoiding wild steps in high curvature via lr_scale.

In practice, on a small MNIST MLP, the gain might be modest or hard to see, and the method can be noisy or “fluctuating” depending on hyperparameters. That’s expected — this is more of an experimental optimizer than a guaranteed win over plain Adam.

# How to Use in Your Own Code

Drop in the A2S class somewhere in your project.

Make sure your model’s parameters are stored in a dictionary like:

model.params = {
    "W1": W1, "b1": b1,
    "W2": W2, "b2": b2,
    ...
}


Your backward pass should return a matching grads dict:

grads = {
    "W1": dW1, "b1": db1,
    "W2": dW2, "b2": db2,
    ...
}


Create the optimizer and call update inside the training loop:

opt = A2S(lr_adam=0.001, lr_sgd=0.01)

for batch in dataloader:
    preds = model.forward(xb)
    grads = model.backward(yb)
    opt.update(model.params, grads)

# Tuning Tips

If training is too noisy / fluctuating:

Lower lr_adam and/or lr_sgd (e.g. lr_adam=5e-4, lr_sgd=5e-3).

Increase curvature_k slightly so high-curvature regions get more damping.

Delay the switch to SGD:

e.g. switch_iter=3000, blend_iters=3000 if your total iters are large.

If SGD seems to never really “take over”:

Decrease switch_iter so the blend starts earlier.

Decrease blend_iters so you move to SGD faster.

Try a slightly higher lr_sgd to make its effect more visible (but this can also increase noise).

# Known Limitations 

This is a toy implementation in plain NumPy, not optimized for speed.

On MNIST with a shallow MLP, Adam is already quite strong; A2S will often match it rather than obviously beat it.

grad_variance is computed but not yet used to modulate behaviour — that’s a natural place for future tweaks (e.g., switching more aggressively to SGD in low-variance regimes).

# License / Attribution

Feel free to reuse or modify this optimizer for experiments or course projects.
If you reference it in a report, you can call it:

A2S (AdaFusion): A curvature-aware Adam–SGD hybrid optimizer.
