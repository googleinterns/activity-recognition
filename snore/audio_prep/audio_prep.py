# This python file is a script to download a given label from AudioSet
# The downloaded audios would be clipped into 10 seconds periods
# Stored in eval_segments, balanced_train, and unblalanced_train separately.

# Usage: python3 audio_prep.py Snoring


from __future__ import unicode_literals
import csv
import json
import youtube_dl
from pydub import AudioSegment
import os
import shutil
from os import listdir
from os.path import isfile, join
import sys

referDoc = "ReferDoc/"

def labelIdSearch(label):
	# Reads in the label, and finds the corresponding id in ontology.json
	# input: str label
	# output: str label_encode
	with open(referDoc+"ontology.json") as f:
		labelInfo = json.load(f)
	for info in labelInfo:
		if info["name"] == label:
			return info["id"]
	return None

def chopAudio(url, destDir, startTime, endTime):
	# Chop the whole audio, 
	# and save only the target part labelled by startTime and endTime.
	# Then remove the original audio.
	# input: str url_from_csv, str destDir, float startTime, float endTime
	# output: None
	onlyfile = [f for f in listdir(destDir+'/tmp') if isfile(join(destDir+'/tmp', f))][0]
	if onlyfile.endswith('.m4a'):
		total = AudioSegment.from_file(destDir+'/tmp/'+url+'.m4a', 'm4a')
	elif onlyfile.endswith('.opus'):
		total = AudioSegment.from_file(destDir+'/tmp/'+url+'.opus', codec='opus')
	else:
		shutil.rmtree(destDir+'/tmp/')
		return None
	sliced = total[startTime*1000: endTime*1000]
	sliced.export(destDir+'/sliced_'+url+'.wav', format='wav')
	shutil.rmtree(destDir+'/tmp/')


def download(labelId, csvFile, destDir):
	# Download all labeled audios from given AudioSet csv list to a dest folder
	# input: str labelId, str csvFile, str destDir
	# output: None
	try:
		os.mkdir(destDir)
	except OSError as error:
		print(error)

	# Extract all labeled audio info from csv
	audio_list = []
	with open(csvFile, newline='') as csvfile:
		info = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in info:
			r = ','.join(row)
			if labelId in r:
				r_lst = r.split(',')
				url, startTime, endTime = r_lst[0], float(r_lst[2]), float(r_lst[4])
				audio_list.append([url, startTime, endTime])

	# Download audio into destDir/tmp/ directory
	# Camera recorded audios would be downloaded in .m4a format
	# Phone recorded audios would be downloaded in .opus format
	for url, startTime, endTime in audio_list:
		ydl_opts = {
			'format': 'bestaudio/best',
			'postprocessors':[{
				'key': 'FFmpegExtractAudio',
				'preferredquality': '192',
			}],
			# Force the file naming of outputs.
			'outtmpl': destDir+'/tmp/'+url+'.%(ext)s'
		}
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			try:
				ydl.download(['https://www.youtube.com/watch?v='+url])
			except:
				print('Downloading Failed.')
				continue
		# Chop the audio in destDir/tmp/ into targeted segments,
		# save the targeted segments in .wav format in destDir/
		chopAudio(url, destDir, startTime, endTime)





if __name__ == "__main__":
	labelId = labelIdSearch(sys.argv[1])
	if labelId == None: raise TypeError
	download(labelId, referDoc+'eval_segments.csv', 'audio_eval')
	download(labelId, referDoc+'balanced_train_segments.csv', 'audio_balanced_train')
	download(labelId, referDoc+'unbalanced_train_segments.csv', 'audio_unbalanced_train')
