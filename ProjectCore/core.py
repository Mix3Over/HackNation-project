from BaseCharts.CAGR import bigCagr
from BaseCharts.Derivative import derivative
from BaseCode.Data_per_sector import CreateDataPerSector


def CreateAndUploadData():
    #CreateDataPerSector(["EN Liczba jednostek gospodarczych"])
    #bigCagr()
    derivative()

CreateAndUploadData()