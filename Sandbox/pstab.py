import numpy as np 


vec = np.array([10.0,100.0,70.0,90.0])

normalized = vec/np.sum(vec)
print(np.sum(vec))
print(vec)
print(normalized)



w = 90
c = 0.0
n=10
d = 0.0

for i in range(n):
	a = np.random.normal(0,1,4)

	b = np.random.randint(0,w)

	h = np.floor((np.dot(a,vec)+b)/w)
	h_u = np.floor((np.dot(a,normalized)+b)/w)
	print("h=")
	print(h)
	print("h_u=")
	print(h_u)
	z = np.sign(h)
	x = np.sign(h_u)
	if z>=0:
		c+=1;
	if w >=0:
		d+=1

print("percentage of non-normalized hash with pos entries :" + str((c/n)*100))

print("percentage of normalized hash with pos entries :" + str((d/n)*100))