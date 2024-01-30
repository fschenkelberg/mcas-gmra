import numpy as np
import matplotlib.pyplot as plt
#Simple script for processing the output of the inverse

def main():
	projections = np.load("mnist_inverse.npy")
	#want to find the highest l2 norm entry
	_, num_pts, num_scales = projections.shape
	count = 0
	#count of well reconstructed pts
	for pt in range(num_pts):
		for scale in range(num_scales):
			embedding = projections[:,pt, scale]
			#check its L2 norm
			if np.linalg.norm(embedding, 2) > .000001:
				count +=1
				pic = np.reshape(embedding,(28,28))
				plt.imshow(pic)
				plt.show()
	print(count)
if __name__ == "__main__":
	main()
