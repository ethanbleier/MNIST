# Ethan Bleier MNIST (tutorial)
# install dependencies: 
# $`!pip install git+https://github.com/tinygrad/tinygrad.git
# 

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from tinygrad import TinyJit

class Model:
	def __init__(self):
		self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
		self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
		self.l3 = nn.Linear(1600, 10)

	def __call__(self, x:Tensor) -> Tensor:
		x = self.l1(x).relu().max_pool2d((2, 2))
		x = self.l2(x).relu().max_pool2d((2, 2))
		return self.l3(x.flatten(1).dropout(0.5))

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = mnist()
	print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
	# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

	'''
	at this step I had to run with:
	$`DISABLE_COMPILER_CACHE=1 python3 model.py`
  to avoid the assertionError: Invalid Metal library. 
	probably due to using conda

	Tried:
	`METAL_XCODE=1 python3 model.py`
	`conda remove --all`
	
	
	only this seemed to work:
	`conda remove conda`
	'''
 
	model = Model()
	acc = (model(X_test).argmax(axis=1) == Y_test).mean()

	print(acc.item())

	optim = nn.optim.Adam(nn.state.get_parameters(model))
	batch_size = 128

	def step():
		Tensor.training = True  # makes dropout work
		samples = Tensor.randint(batch_size, high=X_train.shape[0])
		X, Y = X_train[samples], Y_train[samples]
		optim.zero_grad()
		loss = model(X).sparse_categorical_crossentropy(Y).backward()
		optim.step()
		return loss
	
	# time the step 
  # pretty slow
	# import timeit
	# timeit.repeat(step, repeat=5, number=1)
 
	jit_step = TinyJit(step)

	for step in range(7000):
		loss = jit_step()
		if step % 100 == 0:
			Tensor.training = False
			acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
			print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}")








