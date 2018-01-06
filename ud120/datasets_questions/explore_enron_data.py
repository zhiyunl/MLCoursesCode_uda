#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

file = open("./final_project/final_project_dataset.pkl", "rb")
enron_data = pickle.load(file)


def data_main():
    print(enron_data)
    print("number of dicts(person) is ", len(enron_data))  # number of dicts(person):146 person
    print("number of values(features) of key is ",
          len(enron_data["GLISAN JR BEN F"]))  # number of values(features) of key "GLISAN JR BEN F": 21
    # count poi
    cnt = 0
    for person_name in enron_data:
        if enron_data[person_name]["poi"] is True:
            cnt = cnt + 1
    print("number of poi is ", cnt)  # 18
    # in fact, the poi number should be 35

    print("stock belonging to J P is ", enron_data["PRENTICE JAMES"]["total_stock_value"])  # 1095040
    print("number of emails sent by W C to poi is ", enron_data["COLWELL WESLEY"]['from_this_person_to_poi'])  # 11
    print("value of stock options by J S is ", enron_data["SKILLING JEFFREY K"]['exercised_stock_options'])
    # count salary people
    cnt = 0
    for person_name in enron_data:
        if enron_data[person_name]["salary"] != 'NaN':
            cnt = cnt + 1
    print("number of person have a quantified salary is ", cnt)  # 95
    # count email people
    cnt = 0
    for person_name in enron_data:
        if enron_data[person_name]["email_address"] != 'NaN':
            cnt = cnt + 1
    print("number of person have a email address is ", cnt)  # 111
    # count no total payment people
    cnt = 0
    for person_name in enron_data:
        if enron_data[person_name]["total_payments"] == 'NaN':
            cnt = cnt + 1
    print("number of person have no total payments data is ", cnt)  # 21
    print("the rate of that is ", round(cnt / 146, 3))  # 21
    # count no total payment people in poi
    cnt_poi = 0
    cnt_NaN = 0
    for person_name in enron_data:
        if enron_data[person_name]["poi"] is True:
            cnt_poi = cnt_poi + 1
            if enron_data[person_name]["total_payments"] == 'NaN':
                cnt_NaN = cnt_NaN + 1
    print("in poi, number of person have no total payments data is ", cnt_NaN)  # 0
    print("in poi, the rate of that is ", round(cnt_NaN / cnt_poi, 3))  # 0
