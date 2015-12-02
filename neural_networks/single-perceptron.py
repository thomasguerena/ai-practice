import random

# Basic Neural Network Perceptron
class Perceptron(object):

	# Contructor --
	# 	@param inputs: The number of inputs this perceptron should support.
	#		e.g.    5
	def __init__(self, inputs=0):
		self.weights = []
		for i in range(inputs):
			w = round(random.uniform(-1,1), 10)
			self.weights.append(w)

	# Calculate signal output based on input values and current
	#  weights. A positive output value (+1) indicates a live
	#  signal. A negative output value (-1) indicates no signal.
	#
	#	@param values: An array of integers representing the
	#	  current input values. The number of values must match
	#	  the number of inputs specified in the constructor.
	def pulse(self, values):
		sum = 0
		for i in range(len(values)):
			sum += (values[i] * self.weights[i])
		return self.activate(sum)

	# Determines whether this perceptron fires a signal given the
	#   current input values and weights.
	#
	#	@param output: The dot product of "pulse" values and weights.
	def activate(self, output):
		if output > 0: return 1
		else: return -1

	# Trains this perceptron on some dataset for which the correct
	#   output is known for each input.
	#
	#	@param data: A array of 4-tuples, each containing the three
	#	  input values (x-coord, y-coord, bias) and a desired output.
	#		e.g.    [(4,5,1,1), (9,13,1,-1), ...]
	def train(self, data):
		c = 0.01 # training constant (learning rate)
		for v in data:
			guess = self.pulse(v[:-1])
			error = v[3] - guess

			for i in range(len(self.weights)):
				self.weights[i] += c * error * v[i]


# Example: determine if a point exists above or below
#   some line on an xy-plane.
#
#	Inputs:
#		x-coord
#		y-coord
#		bias
#
#   A "bias" is needed to avoid (0,0) zero sums. A bias
#   always has a value of 1, but its weight can vary.
#
#     XY-Plane:
#
#    0,20                 20,20
#     +-------^-------------+
#	  +       ^             +
#	  +       ^             +
#	  +       ^             +
#	  +  (-1) ^ (+1)        +
#	  +       ^             +
#	  +       ^             +
#	  +       ^             +
#	  +       ^             +
#     +-------^-------------+
#    0,0    (8,0)         20,0
#
#	Any points left of the line x=8 should produce
#	no activation. Any points to the right should
#	activate.

p = Perceptron(3)

print(p.weights) # weights before

# Training
tdata = []
for i in range(1000):
	x = random.randrange(-100, 100)
	y = random.randrange(-100, 100)
	r = 1 if x > 8 else -1
	tdata.append((x,y,1,r))

p.train(tdata)

print(p.weights) # weights after

# Test
print(p.pulse([5,5]))
print(p.pulse([10,10]))