from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile,f_classif
from text_extraction import extract_from_mail,labels_maker

def fun(ngramVal):

    #POPULATING EMAILS AND CORRESPONDING LABELS
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
    print(len(emails))
    print(len(labels))

    #splitting the data set into training and test data
    emails_train,emails_test,labels_train,labels_test = train_test_split(emails,labels,test_size=0.3,random_state=42)

    #testing
    """ flag=CountVectorizer(ngram_range=(3,3),stop_words='english')
    arr=flag.fit_transform(emails_train)
    features=flag.get_feature_names_out()
    print(len(arr.toarray()[10]))
    print(features)
    print(len(features)) """

    #transforming emails into word count
    vectorizer=TfidfVectorizer(ngram_range=(ngramVal,ngramVal),stop_words='english')
    emails_train_transformed=vectorizer.fit_transform(emails_train)
    emails_test_transformed=vectorizer.transform(emails_test)

    #selecting top 10% features else, the dimensionality will be very high
    selector = SelectPercentile(f_classif, percentile=20)
    selector.fit(emails_train_transformed,labels_train)
    emails_train_transformed=selector.transform(emails_train_transformed).toarray()
    emails_test_transformed=selector.transform(emails_test_transformed).toarray()
    return emails_train_transformed,labels_train,emails_test_transformed,labels_test