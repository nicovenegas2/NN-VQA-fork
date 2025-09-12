import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorcircuit as tc
import cotengra
import quimb
from tqdm.notebook import tqdm
from functools import partial

print("############-Import Done-############")
print("##############-Optimizer Setup-############")

optc = cotengra.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel="ray",
    minimize="combo",
    max_time=30,
    max_repeats=1024,
    progbar=True,
)
tc.set_contractor("custom", optimizer=optc, preprocessing=True)

K = tc.set_backend("tensorflow")
tc.set_dtype("complex128")

print("##############-Optimizer Setup Done-############")