import csv

with open('Dane/wsk_fin.xlsx - dane.csv', 'r') as csvfile:
  csv_reader = csv.reader(csvfile)

for row in csv_reader:
    print(row)