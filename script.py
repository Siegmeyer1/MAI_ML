import random

random.seed()
inp = open("salary.csv", "r")
file = open("dataset.csv", "w")
i = 0
while True:
	print("line", i)
	i += 1
	line = inp.readline()
	if not line:
		break
	params = line.split(", ")
	if params[14] == ">50K\n" or params[14] == "salary\n" or random.randint(1, 3) < 2:
		file.write(line)
	else:
		print(params[14])
inp.close()
file.close()
