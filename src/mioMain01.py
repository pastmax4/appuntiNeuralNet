from mieLibrerie import neural01NetLib

print("----Inizio------------")

net =  neural01NetLib.Network([2, 3, 1])
print("----------------------")
print(net.weights)
print(net.weights[1])
print(net.weights[0])

print(net.weights[0][1][0])
print(net.weights[0][1][1])
print("----------------------")
print("----Fine------------")


