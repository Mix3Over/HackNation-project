from data_per_sector import createDataPerSector
from CAGR import bigCagr
from pochodna import pochodna

def createAndUploadData():
    createDataPerSector([])
    #bigCagr()
    pochodna()

createAndUploadData()