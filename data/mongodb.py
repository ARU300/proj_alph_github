import pymongo
from pymongo import MongoClient
import config
import datetime
import bson
import json

from mongoengine import *
from mongoengine import connect

connectionCode = config.mongoclient
connect('pa_userdata', host=connectionCode)


class Post(Document):
    name = StringField(required=True, max_length=20, min_length=2)
    surname = StringField(required=True, max_length=20, min_length=2)
    age = StringField(required=True, max_length=3)
    delta = IntField(required=True, max_length=5, defualt='0')
    unit = StringField(required=True, min_length=1, default='kg')
    published = DateTimeField(default=datetime.datetime.now)


def mongo_post(inpname, inpsurname, inpage, inpdelta, inpunit):
    data = Post(
        name=inpname,
        surname=inpsurname,
        age=inpage,
        delta=inpdelta,
        unit=inpunit,
    )
    data.save()
    print(data.name)


name = input('What is your name?')
surname = input('What is your surname?')
age = input('How old are you?')
delta = input('What is the improvement')
unit = input('What unit was this measured in?')
mongo_post(inpname=name, inpsurname=name, inpdelta=delta, inpage=age, inpunit=unit)

