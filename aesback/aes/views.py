from django.shortcuts import render
from django.http import HttpResponse
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen
from django.views.decorators.csrf import csrf_exempt
from .pretrain import PretrainedModel as PM

def getsource(id):
	url = "https://stackoverflow.com/questions/"+str(id)
	html = urlopen(url).read().decode('utf-8')
	soup = BeautifulSoup(html, features='lxml')
	title = soup.find('h1',itemprop="name").text
	text = soup.find('div',itemprop="text").text.replace('\n','<br/>')
	content_tags = soup.find('div',itemprop="text").find_all('p')
	content = ""
	for c in content_tags:
		content+=c.text
	code_tags = soup.find('div',itemprop="text").find_all('pre')
	code = ""
	for c in code_tags:
		code+=c.text
	tags = soup.find('div',class_="post-taglist").find_all('a',class_="post-tag")
	tag = ""
	for c in tags:
		tag = tag+c.text+' '
	data = {'title':title,'text':text,'content':content,'code':code,'tag':tag.strip()}
	return data

@csrf_exempt
def index(request):
	ori = getsource(request.POST.get('id'))
	res = PM.predict(ori)
	return HttpResponse(json.dumps({'ori':ori,'res':res}), content_type="application/json")
