import numpy as np
from botor import run_parameter_range

parameters = {
	"mu_bin": [0.4],
	"e_bin": [0.04],
	"mu_p": [0.5],
	"mass_p": [0.0001],
	"e_p": np.linspace(0.0,0.9,10),
	"phi_bin": [np.pi*0],
	"inc": [np.pi*0],
	"Omega": [np.pi*0],
	"per": [np.pi*0.5],
	"phi_p": [np.pi*1],
	"aq_p": np.linspace(3.0,6.0,31),
	"a/q": "a",
	"mode": "2P"
}

run_parameter_range(parameters, "/home/stefan/Documents/1to1circumbinary/example/data/", "/home/stefan/Documents/1to1circumbinary/example/output/example_0.out", export = None, dt = 10, run = "linear")