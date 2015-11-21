import time
import math
import random

################################### Function Definitions ###########################################

def norm(x): # Calculate the norm of the document vector x
    sos = 0
    for i in x:
        sos += pow(x[i],2) # sum of squares
    return(math.sqrt(sos))

def dist(x,y,normx,normy): # Measure the distance between the document vectors x and y
    S = 0
    for i in x:
        try:
            S += x[i]*y[i]
        except KeyError:
            continue
    cosine = S/(normx*normy)
    return(1-cosine)

def add(x,y): # add the document vectors x and y
    s = x.copy()
    for i in y:
        try:
            s[i] += y[i]
        except KeyError:
            s[i] = y[i]
    return(s)

def k_means(k,d): # Function for K-means clustering of documents using set of document vectors d
    threshold = 0.0001
    init = random.sample(range(1,N+1),k) # initial choice of centriods
    l1 = "Initial choice of centroids :: " + str(init)
    print(l1)
    centroid = []
    norm_c = []
    for c in init:
        centroid.append(d[c])
        norm_c.append(norm_d[c])
    iter_no = 0
    sd1 = 0
    l2 = ""
    while 1:
        cluster = [[] for i in range(k)]
        radius = [0]*k
        s = [{} for i in range(k)]
        n = [0]*k
        sd = 0
        for i in range(1,N+1):
            m = 1
            j = random.randint(0,k-1) # If distance of a document from each cluster is 1, choose the cluster randomly
            for c in range(k):
                distance = dist(d[i],centroid[c],norm_d[i],norm_c[c])
                if distance < m:
                    j = c
                    m = distance
            cluster[j].append(i)
            if m > radius[j]:
                radius[j] = m
            sd += m # sum of distances from centroids (objective function to be minimized)
            s[j] = add(s[j],d[i])
            n[j] += 1
        for i in range(k):
            for x in s[i]:
                s[i][x] = s[i][x]/n[i]
            centroid[i] = s[i].copy()
            norm_c[i] = norm(centroid[i])
        print(iter_no,sd)
        if abs(sd1-sd)/sd < threshold:
            break
        else:
            if iter_no == 30:
                l2 = "Can't hit convergence after "
                break
            sd1 = sd
            iter_no += 1
            continue
    l2 += "Number of iterations :: " + str(iter_no)
    print(l2)
    l3 = "Cluster sizes :: " + str(n)
    print(l3)
    l4 = "Radius :: " + str(radius)
    print(l4)
    z = l1 + "\n" + l2 + "\n" + l3 + "\n" + l4 + "\n"
    return(z)

##################### Input Processing & Constructing Document Vectors #############################

print("Data sources :: kos, nips, enron")
source = str(input("Enter the data source :: "))
Start = time.time()
f = open("docword." + source + ".txt",'r')
N = int(f.readline().strip()) # No. of documents in the collection
W = int(f.readline().strip()) # No. of words in the vocabulary
NZW = int(f.readline().strip()) # No. of non-zero frequency entries for this collection
d = {}
df = {}
while 1:
    l = f.readline()
    if not l: break
    else:
        l = l.strip().split(' ')
        for j in range(len(l)):
            l[j] = int(l[j])
        try:
            d[l[0]][l[1]] = l[2]
        except KeyError:
            d[l[0]] = {}
            d[l[0]][l[1]] = l[2]
        try:
            df[l[1]] += 1
        except KeyError:
            df[l[1]] = 1.0
f.close()
idf = {}
for j in df:
    idf[j] = math.log(N/df[j]) # inverse document frequency of wordID j
norm_d = {}
for i in d:
    m = max(d[i].values())
    for j in d[i]:
        tf = float(d[i][j])/m # term frequency of wordID j for docID i
        d[i][j] = tf*idf[j] # TF-IDF weight corresponding to the docID i and wordID j
    norm_d[i] = norm(d[i])
Stop = time.time()
t0 = Stop-Start
l0 = "Time for input processing & constructing document vectors :: " + str(t0)
print(l0)

####################################### K-means Clustering #########################################

t = t0
while 1:
    output = l0 + "\n"
    k = int(input("Enter the value of k (Try values between 3 to 10) :: "))
    for i in range(5):
        l5 = "\nInitial choice # " + str(i+1)
        print(l5)
        start = time.time()
        z = k_means(k,d)
        stop = time.time()
        t1 = stop-start
        t += t1
        l6 = "Runtime :: " + str(t1)
        print(l6)
        output += l5 + "\n" + z + l6 + "\n"
    f = open("Output/output." + source + "." + str(k) + ".clusters.txt",'w')
    f.write(output)
    f.close()
    while 1:
        flag = input("\nEnter 0 to exit, 1 to continue with the same dataset :: ")
        if flag == '0' or flag == '1':
            break
        else:
            print("Wrong input. Please enter correct input.")
            continue
    if flag == '0':
        break
    elif flag == '1':
        continue
print("Total Runtime :: ",t,"\n")
