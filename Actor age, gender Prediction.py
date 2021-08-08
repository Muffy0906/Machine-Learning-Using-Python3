#A program in python to find 10 most popular Bollywood actor . Extract the data from IMDB . Find their age and group them into three age groups (0-30 ,31-60,61-100) . Also create separate group according to their Gender , Race and Emotion

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import urllib.request
from deepface import DeepFace
import cv2

html = urlopen('https://www.imdb.com/poll/pZSc95Bt5II/results')
bs = BeautifulSoup(html, 'html.parser')
imgsrc = []
images = bs.find_all('img', {'src': re.compile('.jpg')})
n = []
names = []
namecontainers = bs.find_all('h3')
x = str(namecontainers[1])
ext = '.jpg'

for name in namecontainers:
    x = str(name)
    n.append(x[31:-9])

for i in range(10):
    names.append(n[i+1] + ext)

for image in images:
    x = image['src'] + '\n'
    imgsrc.append(x)

for i in range(10):
    url = imgsrc[i]
    urllib.request.urlretrieve(url, '/home/muffy/Documents/Images/%s ' % names[i])


def age(m):
    img = cv2.imread('/home/muffy/Documents/Images/%s ' % m)
    result = DeepFace.analyze(img, actions=['age'])
    return result


ag = []
a = []
b = []
path = '~/Documents/Images/'
for i in range(10):
    a.append(age(names[i]))
for x in a:
    b.append(str(x))
for x in b:
    ag.append(int(x[8:-1]))
print("\nGrouping based on age:")
print("\nAge Group 1: (0-30)")
for i in range(10):
    if 0 <= ag[i] <= 30:
        print("\t", n[i+1])
        print("\t", "Age: ", ag[i])

print("\nAge Group 2: (31-60)")
for i in range(10):
    if 31 <= ag[i] <= 60:
        print("\t", n[i+1])
        print("\t", "Age: ", ag[i])

print("\nAge Group 3: (61-100)")
for i in range(10):
    if 61 <= ag[i] <= 100:
        print("\t", n[i+1])
        print("\t", "Age: ", ag[i])


def gender(m):
    img = cv2.imread('/home/muffy/Documents/Images/%s ' % m)
    result = DeepFace.analyze(img, actions=['gender'])
    return result['gender']


gend = []
for i in range(10):
    gend.append(gender(names[i]))

print("\n\nGrouping based on gender:")
print("\nFemales:")
for i in range(10):
    if gend[i] == 'Woman':
        print("\t", n[i+1])

print("\nMales:")
for i in range(10):
    if gend[i] == 'Man':
        print("\t", n[i+1])


def emotion(m):
    img = cv2.imread('/home/muffy/Documents/Images/%s ' % m)
    result = DeepFace.analyze(img, actions=['emotion'])
    return result['dominant_emotion']


e = []
for i in range(10):
    e.append(emotion(names[i]))

etype = ['sad']
for i in e:
    if i in etype:
        continue
    else:
        etype.append(i)

print("\n\nGrouping based on emotion:")
for x in range(len(etype)):
    print("\n",etype[x], "ones:")
    for i in range(10):
        if e[i] == etype[x]:
            print("\t", n[i+1])


def race(m):
    img = cv2.imread('/home/muffy/Documents/Images/%s ' % m)
    result = DeepFace.analyze(img, actions=['race'])
    return result['dominant_race']


r = []
for i in range(10):
    r.append(race(names[i]))

rtype = ['latino hispanic']
for i in r:
    if i in rtype:
        continue
    else:
        rtype.append(i)

print("\n\nGrouping based on race:")
for x in range(len(rtype)):
    print("\n",rtype[x], "ones:")
    for i in range(10):
        if r[i] == rtype[x]:
            print("\t", n[i+1])