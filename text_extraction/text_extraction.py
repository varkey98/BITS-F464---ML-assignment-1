print('beginning')
# import pdfplumber

""" def extract_from_mail(fileName):
    pdf =pdfplumber.open(fileName)
    message=''
    for i in range(len(pdf.pages)):
        curr_page=pdf.pages[i]
        message+=curr_page.extract_text()
    return message """

def extract_from_mail(message):
    key="-------------------------"
    mail=open(message,"r")
    messages=mail.readlines()
    ret=[]
    str=''
    for line in messages:
        #print(line)
        if line.rstrip() == key:
            print('flag')
            ret.append(str)
            str=''
        elif ':' not in line:
            str+=line
    mail.close()
    return ret

ls=extract_from_mail('messages.txt')
print(len(ls))
for msg in ls:
    print(msg)
#print(extract_from_mail('mail.pdf'))
print('exit')