def extract_from_mail(message):
    key="-------------------------"
    mail=open(message,"r",encoding='utf-8')
    messages=mail.readlines()
    ret=[]
    str=''
    for line in messages:
        #print(line)
        if line.rstrip() == key:
            ret.append(str)
            str=''
        elif ':' not in line:
            str+=line
    mail.close()
    return ret

def labels_maker(label,n):
    ret=[label]*n
    return ret
import csv

emails=extract_from_mail('placements.txt')
n=len(emails)
labels=labels_maker('placements',n)
emails.extend(extract_from_mail('promotions.txt'))
labels.extend(labels_maker('promotions',len(emails)-n))
n=len(emails)
emails.extend(extract_from_mail('social.txt'))
labels.extend(labels_maker('social',len(emails)-n))
n=len(emails)
emails.extend(extract_from_mail('news.txt'))
labels.extend(labels_maker('news',len(emails)-n))
n=len(emails)
emails.extend(extract_from_mail('acads.txt'))
labels.extend(labels_maker('acads',len(emails)-n))
n=len(emails)
emails.extend(extract_from_mail('misc.txt'))
labels.extend(labels_maker('misc',len(emails)-n))

finalList=[]
for i in range(len(emails)):
    finalList.append([emails[i],labels[i]])
fields=['Emails','Labels']
filename="dataset.csv"
with open (filename,"w",newline='',encoding='utf-8') as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(finalList)
finalList=[]
with open(filename,'r',encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        finalList.append(row)
emails=[]
labels=[]
for row in finalList:
    emails.append(row[0])
    labels.append(row[1])
print(emails[0])
print(labels[0])
print(len(emails))
print(len(labels))